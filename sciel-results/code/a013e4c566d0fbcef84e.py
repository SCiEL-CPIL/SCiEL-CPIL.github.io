import time
import subprocess
import sys
import os
import tempfile

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # C++ Code for Fast Simulated Annealing
    # Uses bitsets for fast triangle checking.
    # Input: n, time_limit
    # Output: n, colors (space separated)
    cpp_code = r'''
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <numeric>
#include <cmath>

using namespace std;

// Global state
int N;
vector<int> colors; // length N*(N-1)/2
vector<int> adj; // length N*N, adj[i*N+j] = color
int total_edges;

// Bitsets for fast triangle checking
// For each vertex i and color c, a bitset of neighbors j such that edge (i,j) is color c.
// vector<array<unsigned long long, K>> neighbors; 
// K = ceil(N/64)
int K;
vector<vector<unsigned long long>> neigh; // neigh[i][c][block]

// Count of monochromatic triangles
// We can maintain this incrementally, but for simplicity in SA, 
// we can recompute or use a faster check.
// Actually, recomputing total mono triangles is O(N^3/64) which is fine for N=50.
// 50^3/64 ~ 2000 ops. Very fast.

long long count_mono_triangles() {
    long long count = 0;
    for (int i = 0; i < N; ++i) {
        for (int c = 0; c < 4; ++c) {
            // For each pair (j, k) in neighbors of i with color c, if edge (j,k) is also c, count 1.
            // To avoid double counting, we can just sum over i<j<k.
            // Or simpler: for each i, for each c, count pairs (j,k) in neigh[i][c] such that edge(j,k)==c.
            // Sum over all i, divide by 3? No, each triangle (i,j,k) is counted 3 times (once for each vertex).
            // So count = (sum over i of pairs) / 3.
            
            // Efficiently count pairs in neigh[i][c] that are also connected by color c.
            // This is equivalent to counting edges in the subgraph induced by neigh[i][c] with color c.
            // Let S = neigh[i][c]. We want to count edges (j,k) in S such that adj[j,k]==c.
            // We can iterate j in S, and for each j, iterate k in S with k>j, check adj[j,k].
            // This is O(|S|^2). Sum over i is O(N * N^2) = O(N^3).
            
            // Let's optimize:
            // For a fixed i and c, we have a bitset S.
            // We want to count edges in the graph G_c induced by S.
            // We can do this by iterating j in S, and for each j, looking at neigh[j][c] & S, and counting bits.
            // But we must avoid double counting (j,k) and (k,j).
            // If we sum popcount(neigh[j][c] & S) for all j in S, we count each edge twice.
            // So count_for_i_c = (1/2) * sum_{j in S} popcount(neigh[j][c] & S).
            
            int sum = 0;
            // Iterate j in S
            for (int j = 0; j < N; ++j) {
                if (j == i) continue;
                // Check if j is in S (i.e., edge(i,j) is color c)
                if (adj[i*N+j] == c) {
                    // Count neighbors k of j with color c that are also in S (k != i, k != j)
                    // neigh[j][c] contains i. We want to intersect with S.
                    // S is neigh[i][c].
                    // Intersection: neigh[j][c] & neigh[i][c].
                    // This intersection contains i (since i in neigh[j][c] and i in neigh[i][c]? No, i is not in neigh[i][c]).
                    // Wait, neigh[i][c] contains j. neigh[j][c] contains i.
                    // The intersection contains vertices k such that edge(i,k)==c and edge(j,k)==c.
                    // This is exactly the set of k that form a triangle (i,j,k) with color c.
                    // So popcount(neigh[j][c] & neigh[i][c]) gives the number of such k.
                    // We sum this over all j in S.
                    // Each triangle (i,j,k) is counted twice: once when j is iterated (counts k) and once when k is iterated (counts j).
                    // So summing over j in S gives 2 * (# triangles involving i and color c).
                    int cnt = 0;
                    for (int b = 0; b < K; ++b) {
                        cnt += __builtin_popcountll(neigh[j][c][b] & neigh[i][c][b]);
                    }
                    sum += cnt;
                }
            }
            count += sum / 2;
        }
    }
    // Each triangle (i,j,k) is counted 3 times in the total sum (once for each vertex i, j, k).
    // Wait, my loop `for i` counts triangles where i is the "center".
    // A triangle (u,v,w) is counted:
    // - when i=u: v and w are in S_u. The pair (v,w) is counted in the sum for u.
    // - when i=v: u and w are in S_v. The pair (u,w) is counted in the sum for v.
    // - when i=w: u and v are in S_w. The pair (u,v) is counted in the sum for w.
    // So yes, total count is 3 * (# mono triangles).
    return count / 3;
}

void update_adj() {
    // Rebuild adj from colors
    int idx = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = i+1; j < N; ++j) {
            int c = colors[idx++];
            adj[i*N+j] = c;
            adj[j*N+i] = c;
        }
    }
}

void build_neigh() {
    // Rebuild neigh bitsets
    // Initialize to 0
    for (int i = 0; i < N; ++i) {
        for (int c = 0; c < 4; ++c) {
            for (int b = 0; b < K; ++b) {
                neigh[i][c][b] = 0;
            }
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            int c = adj[i*N+j];
            neigh[i][c][j/64] |= (1ULL << (j%64));
        }
    }
}

int get_edge_idx(int i, int j) {
    if (i > j) swap(i, j);
    // Sum of 1..i-1 is i*(i-1)/2
    return i*(i-1)/2 + (j - i - 1);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "Usage: prog n time_limit" << endl;
        return 1;
    }
    N = atoi(argv[1]);
    double time_limit = atof(argv[2]);
    
    total_edges = N*(N-1)/2;
    K = (N + 63) / 64;
    
    // Initialize colors with 32-vertex seed embedded in N
    // If N > 32, we need to fill the rest.
    // Strategy: Start with a valid 32-coloring, extend greedily to N, then SA.
    
    // 1. Generate 32-vertex coloring using GF(16)
    vector<int> base32;
    for (int i = 0; i < 32; ++i) {
        for (int j = i+1; j < 32; ++j) {
            int c;
            if (i/16 == j/16) {
                // Inside block
                int a = i % 16;
                int b = j % 16;
                // Log table for GF(16)
                // We need a fast way to compute log(a^b) mod 3.
                // Precompute log table for GF(16)
                // Let's hardcode the log table for GF(16) with primitive polynomial x^4+x+1 (0x13)
                // Elements: 1, 2, 4, 8, 3, 6, 12, 11, 15, 13, 7, 14, 10, 5, 1
                // Indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
                // We need log(x/y).
                // Since we only have 16 elements, we can precompute a 16x16 table of colors.
                // But for speed in C++, let's just compute it once.
                static int log_table[16];
                static int pow_table[16];
                static bool init = false;
                if (!init) {
                    int x = 1;
                    for (int e = 0; e < 15; ++e) {
                        log_table[x] = e;
                        pow_table[e] = x;
                        // x = x * 2 in GF(16)
                        // x = x << 1
                        // if x >= 16, x ^= 0x13 (11)
                        x = x << 1;
                        if (x & 16) x ^= 11;
                    }
                    init = true;
                }
                if (a == 0 || b == 0) {
                    // This case doesn't happen in 1..15 if we map 0..15 to field elements?
                    // The seed code uses i%16. If i%16 is 0, it's the zero element.
                    // The seed code: log[(i % 16) ^ (j % 16)] % 3.
                    // If i%16 == j%16, XOR is 0. log[0] is undefined in the loop (e=0..14).
                    // Wait, the seed code has `log[x] = e` for e in 0..14.
                    // If XOR is 0, it's not in the table.
                    // Let's check the seed code logic again.
                    // `get_16_color(i, j): return log[(i % 16) ^ (j % 16)] % 3`
                    // If i%16 == j%16, XOR is 0.
                    // In the seed, `log` is a dict. If 0 is not in log, it crashes?
                    // No, the seed says `for i in range(32): for j in range(i+1, 32)`.
                    // If i//16 == j//16, then i and j are in the same block of 16.
                    // i != j, so i%16 != j%16. So XOR is never 0.
                    // So we are safe.
                }
                int val = a ^ b;
                c = log_table[val] % 3;
            } else {
                c = 3;
            }
            base32.push_back(c);
        }
    }
    
    // Initialize colors vector of size total_edges
    colors.resize(total_edges);
    // Copy base32 to first 32 vertices
    int idx32 = 0;
    for (int i = 0; i < 32; ++i) {
        for (int j = i+1; j < 32; ++j) {
            colors[get_edge_idx(i, j)] = base32[idx32++];
        }
    }
    
    // Fill rest with random colors (0-3)
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 3);
    for (int i = 0; i < total_edges; ++i) {
        // Check if this edge is in the first 32 vertices
        // We can just check if i < base32.size() and it's not already set?
        // Easier: iterate all edges, if not in first 32 block, randomize.
    }
    // Let's just randomize all edges not in the 32x32 subgraph.
    for (int i = 0; i < N; ++i) {
        for (int j = i+1; j < N; ++j) {
            if (i < 32 && j < 32) {
                // Already set
                continue;
            }
            colors[get_edge_idx(i, j)] = dis(gen);
        }
    }
    
    update_adj();
    build_neigh();
    
    long long current_energy = count_mono_triangles();
    long long best_energy = current_energy;
    vector<int> best_colors = colors;
    
    auto start = chrono::high_resolution_clock::now();
    
    double T = 10.0;
    const double cooling_rate = 0.999;
    
    while (true) {
        auto now = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration_cast<chrono::milliseconds>(now - start).count() / 1000.0;
        if (elapsed > time_limit - 1.0) break;
        
        // Pick random edge
        int e_idx = gen() % total_edges;
        int old_c = colors[e_idx];
        int new_c = dis(gen);
        if (old_c == new_c) continue;
        
        // Find i, j from e_idx
        // i*(i-1)/2 + (j-i-1) = e_idx
        // Approx i = sqrt(2*e_idx)
        int i = (int)sqrt(2.0 * e_idx);
        while (i*(i-1)/2 + (i - i - 1) <= e_idx) {
             // This loop is tricky. Let's just use a lookup table or binary search.
             // For N=50, total edges = 1225.
             // We can precompute edge_to_ij.
             break;
        }
        // Let's precompute edge_to_i, edge_to_j
        vector<int> edge_i(total_edges), edge_j(total_edges);
        int idx = 0;
        for (int u = 0; u < N; ++u) {
            for (int v = u+1; v < N; ++v) {
                edge_i[idx] = u;
                edge_j[idx] = v;
                idx++;
            }
        }
        
        i = edge_i[e_idx];
        int j = edge_j[e_idx];
        
        // Update colors
        colors[e_idx] = new_c;
        adj[i*N+j] = new_c;
        adj[j*N+i] = new_c;
        
        // Update neigh bitsets
        // Remove old_c from neigh[i][old_c] and neigh[j][old_c]
        neigh[i][old_c][j/64] &= ~(1ULL << (j%64));
        neigh[j][old_c][i/64] &= ~(1ULL << (i%64));
        
        // Add new_c to neigh[i][new_c] and neigh[j][new_c]
        neigh[i][new_c][j/64] |= (1ULL << (j%64));
        neigh[j][new_c][i/64] |= (1ULL << (i%64));
        
        long long new_energy = count_mono_triangles();
        long long delta = new_energy - current_energy;
        
        if (delta < 0 || (double)gen() / (double)mt19937::max() < exp(-delta / T)) {
            current_energy = new_energy;
            if (current_energy < best_energy) {
                best_energy = current_energy;
                best_colors = colors;
            }
        } else {
            // Revert
            colors[e_idx] = old_c;
            adj[i*N+j] = old_c;
            adj[j*N+i] = old_c;
            
            neigh[i][new_c][j/64] &= ~(1ULL << (j%64));
            neigh[j][new_c][i/64] &= ~(1ULL << (i%64));
            
            neigh[i][old_c][j/64] |= (1ULL << (j%64));
            neigh[j][old_c][i/64] |= (1ULL << (i%64));
        }
        
        T *= cooling_rate;
        if (T < 0.01) T = 0.01;
    }
    
    // Output best result
    // If best_energy > 0, it's invalid. But we want to maximize n.
    // The task says "No colour class may contain a triangle".
    // So we must return a valid coloring.
    // If best_energy > 0, we should try to find a valid one or return the 32-vertex one.
    // However, the evaluator measures n. If we return n=50 with violations, is it invalid?
    // "An invalid result ... scores 0."
    // So we MUST return a valid coloring.
    
    // If best_energy > 0, we can try to repair or just fallback to 32.
    // But maybe we found a valid one?
    if (best_energy == 0) {
        cout << N << " ";
        for (int c : best_colors) {
            cout << c << " ";
        }
    } else {
        // Fallback to 32-vertex valid coloring
        cout << 32 << " ";
        for (int c : base32) {
            cout << c << " ";
        }
    }
    cout << endl;
    return 0;
}
'''
    
    # Write C++ code to temp file
    with tempfile.NamedTemporaryFile(suffix='.cpp', delete=False, mode='w') as f:
        f.write(cpp_code)
        cpp_file = f.name
    
    # Compile
    exe_file = cpp_file.replace('.cpp', '')
    try:
        compile_cmd = ['g++', '-O3', '-o', exe_file, cpp_file]
        subprocess.run(compile_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        # If compilation fails, return seed 32
        os.unlink(cpp_file)
        if os.path.exists(exe_file): os.unlink(exe_file)
        
        def make_32_coloring():
            colors = []
            def mul(a, b):
                r = 0
                for i in range(4):
                    if b >> i & 1:
                        r ^= a << i
                for i in range(7, 3, -1):
                    if r >> i & 1:
                        r ^= 0b10011 << (i - 4)
                return r
            log, x = {}, 1
            for e in range(15):
                log[x] = e
                x = mul(x, 2)
            def get_16_color(i, j):
                return log[(i % 16) ^ (j % 16)] % 3
            for i in range(32):
                for j in range(i + 1, 32):
                    if i // 16 == j // 16:
                        colors.append(get_16_color(i, j))
                    else:
                        colors.append(3)
            return colors
        return {'n': 32, 'colors': make_32_coloring()}

    # Run C++ program
    # We want to try n=50.
    # Time limit for C++ is 100s (leaving 10s for python overhead)
    time_limit = 100.0
    try:
        # Run with n=50
        result = subprocess.run([exe_file, '50', str(time_limit)], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                timeout=110)
        output = result.stdout.decode().strip()
        
        if output:
            parts = output.split()
            n_found = int(parts[0])
            colors_found = [int(x) for x in parts[1:1+n_found*(n_found-1)//2]]
            
            # Validate in Python just in case
            if len(colors_found) == n_found*(n_found-1)//2:
                # Check validity
                adj = [[-1]*n_found for _ in range(n_found)]
                idx = 0
                for i in range(n_found):
                    for j in range(i+1, n_found):
                        adj[i][j] = colors_found[idx]
                        adj[j][i] = colors_found[idx]
                        idx += 1
                valid = True
                for i in range(n_found):
                    for j in range(i+1, n_found):
                        c = adj[i][j]
                        for k in range(j+1, n_found):
                            if adj[i][k] == c and adj[j][k] == c:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    return {'n': n_found, 'colors': colors_found}
                else:
                    # If invalid, fallback
                    pass
            
            # Try n=40 if 50 failed or was invalid
            if n_found < 50 or not valid:
                result = subprocess.run([exe_file, '40', str(time_limit)], 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE, 
                                        timeout=110)
                output = result.stdout.decode().strip()
                if output:
                    parts = output.split()
                    n_found = int(parts[0])
                    colors_found = [int(x) for x in parts[1:1+n_found*(n_found-1)//2]]
                    if len(colors_found) == n_found*(n_found-1)//2:
                        adj = [[-1]*n_found for _ in range(n_found)]
                        idx = 0
                        for i in range(n_found):
                            for j in range(i+1, n_found):
                                adj[i][j] = colors_found[idx]
                                adj[j][i] = colors_found[idx]
                                idx += 1
                        valid = True
                        for i in range(n_found):
                            for j in range(i+1, n_found):
                                c = adj[i][j]
                                for k in range(j+1, n_found):
                                    if adj[i][k] == c and adj[j][k] == c:
                                        valid = False
                                        break
                                if not valid: break
                            if not valid: break
                        if valid:
                            return {'n': n_found, 'colors': colors_found}
            
    except Exception as e:
        pass
    finally:
        os.unlink(cpp_file)
        if os.path.exists(exe_file): os.unlink(exe_file)

    # Fallback to 32
    def make_32_coloring():
        colors = []
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        def get_16_color(i, j):
            return log[(i % 16) ^ (j % 16)] % 3
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(get_16_color(i, j))
                else:
                    colors.append(3)
        return colors
    return {'n': 32, 'colors': make_32_coloring()}
