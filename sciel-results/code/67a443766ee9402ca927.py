"""Pancyclic graph on 40 vertices - structural approach with greedy cycle extraction."""
import time
import random
from itertools import combinations

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 75.0  # safety margin before 90s kill

    def norm(e):
        a, b = e
        return (a, b) if a < b else (b, a)

    def build_adjacency(edges):
        adj = [set() for _ in range(N)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def verify_cycles(edges, cycles):
        """Verify all cycles are valid (correct length, distinct vertices, edges exist)."""
        adj = build_adjacency(edges)
        for L in range(3, N + 1):
            if L not in cycles:
                return False
            c = cycles[L]
            if len(c) != L:
                return False
            if len(set(c)) != L:
                return False
            for i in range(L):
                u = c[i]
                v = c[(i + 1) % L]
                if v not in adj[u]:
                    return False
        return True

    def find_cycle_length(L, adj, edges):
        """Find a cycle of exactly length L using DFS with pruning."""
        # Use backtracking with careful pruning
        found = [None]
        
        def dfs(path, visited):
            if found[0]:
                return
            if time.time() - start_time > deadline - 5:
                found[0] = "timeout"
                return
            if len(path) == L:
                if path[0] in adj[path[-1]]:
                    found[0] = list(path)
                return
            if len(path) > L:
                return
            
            last = path[-1]
            for nxt in adj[last]:
                if nxt == path[0] and len(path) < L:
                    continue
                if nxt in visited:
                    continue
                # Pruning: check if remaining vertices can complete the cycle
                remaining = L - len(path)
                if remaining < 1:
                    continue
                path.append(nxt)
                visited.add(nxt)
                dfs(path, visited)
                path.pop()
                visited.remove(nxt)
                if found[0]:
                    return
        
        # Try from vertex 0 first (most likely to have many connections)
        dfs([0], {0})
        return found[0]

    def find_cycle_length_fast(L, adj):
        """Faster cycle finding using heuristic ordering."""
        found = [None]
        
        # Order neighbors by degree (higher degree first) for better pruning
        def dfs(path, visited, depth):
            if found[0]:
                return
            if time.time() - start_time > deadline - 5:
                found[0] = "timeout"
                return
            if depth == L:
                if path[0] in adj[path[-1]]:
                    found[0] = list(path)
                return
            
            last = path[-1]
            # Try neighbors in order of decreasing degree
            neighbors = sorted(adj[last], key=lambda x: len(adj[x]), reverse=True)
            for nxt in neighbors:
                if nxt == path[0] and depth < L:
                    continue
                if nxt in visited:
                    continue
                path.append(nxt)
                visited.add(nxt)
                dfs(path, visited, depth + 1)
                path.pop()
                visited.remove(nxt)
                if found[0]:
                    return
        
        dfs([0], {0}, 1)
        if found[0]:
            return found[0]
        
        # Try from other starting vertices
        for start in range(1, N):
            if time.time() - start_time > deadline - 5:
                return None
            dfs([start], {start}, 1)
            if found[0]:
                return found[0]
        return None

    # ---------- Build initial graph: Hamiltonian cycle + strategic chords ----------
    # Hamiltonian cycle: 0-1-2-...-39-0
    base_edges = set()
    for i in range(N):
        base_edges.add(norm((i, (i + 1) % N)))
    
    # Add chords from vertex 0 to cover cycle lengths
    # Chord (0, k) creates cycles of length k+1 and 41-k (for k in 2..20)
    # We need to cover lengths 3..39
    chords_to_add = []
    for k in range(2, 21):
        chords_to_add.append(norm((0, k)))
    
    all_edges = set(base_edges)
    for ch in chords_to_add:
        all_edges.add(ch)
    
    # Extract cycles for initial graph
    cycles = {}
    cycles[40] = list(range(N))  # Hamiltonian cycle
    
    adj = build_adjacency(all_edges)
    
    for L in range(3, 40):
        if L <= 21:
            # Use chord (0, L-1), forward arc: 0, 1, 2, ..., L-1
            k = L - 1
            cycle = list(range(0, k + 1))
        else:
            # Use chord (0, 41-L), backward arc: 0, 39, 38, ..., 41-L
            k = 41 - L
            cycle = [0] + list(range(39, k - 1, -1))
        cycles[L] = cycle
    
    # Verify initial construction
    if not verify_cycles(all_edges, cycles):
        # Shouldn't happen, but try to fix
        for L in range(3, N + 1):
            if L in cycles:
                c = cycles[L]
                if len(c) == L and len(set(c)) == L:
                    valid = True
                    for i in range(L):
                        u = c[i]
                        v = c[(i + 1) % L]
                        if v not in adj[u]:
                            valid = False
                            break
                    if not valid:
                        found = find_cycle_length(L, adj, all_edges)
                        if found and found != "timeout":
                            cycles[L] = found
    
    # ---------- Edge deletion phase: try to remove edges while maintaining pancyclicity ----------
    # Sort edges: Hamiltonian cycle edges first (more critical), then chords
    h_cycle_edges = set(norm((i, (i + 1) % N)) for i in range(N))
    chord_edges = [e for e in all_edges if e not in h_cycle_edges]
    
    # Shuffle chords for randomized deletion order
    random.seed(42)
    random.shuffle(chord_edges)
    
    # Try deleting each chord
    for chord in chord_edges:
        if time.time() - start_time > deadline - 10:
            break
        if chord not in all_edges:
            continue
        
        test_edges = set(all_edges)
        test_edges.discard(chord)
        
        # Check if graph is still pancyclic
        test_adj = build_adjacency(test_edges)
        still_pancyclic = True
        
        # First check if our known cycles are still valid
        for L in range(3, N + 1):
            c = cycles.get(L)
            if c is None:
                still_pancyclic = False
                break
            ok = True
            for i in range(L):
                u = c[i]
                v = c[(i + 1) % L]
                if v not in test_adj[u]:
                    ok = False
                    break
            if not ok:
                # Try to find a replacement cycle
                found = find_cycle_length_fast(L, test_adj)
                if not found:
                    still_pancyclic = False
                    break
        
        if still_pancyclic:
            all_edges = test_edges
            adj = test_adj
    
    # ---------- Second phase: try deleting some Hamiltonian cycle edges ----------
    # This is riskier but could yield further reduction
    h_edges_list = sorted(h_cycle_edges)
    random.shuffle(h_edges_list)
    
    for he in h_edges_list:
        if time.time() - start_time > deadline - 5:
            break
        if he not in all_edges:
            continue
        
        test_edges = set(all_edges)
        test_edges.discard(he)
        
        # Need Hamiltonian cycle check
        test_adj = build_adjacency(test_edges)
        
        # Check if a Hamiltonian cycle exists (length 40)
        # This is NP-hard in general, but for our sparse graph, try the original one first
        h_cycle_valid = True
        for i in range(N):
            u = i
            v = (i + 1) % N
            if v not in test_adj[u]:
                h_cycle_valid = False
                break
        
        if not h_cycle_valid:
            # Try to find any Hamiltonian cycle (expensive, only try if quick)
            if time.time() - start_time < deadline - 3:
                # Simple backtracking for Hamiltonian cycle
                h_found = [False]
                def hdfs(path, visited):
                    if h_found[0]:
                        return
                    if time.time() - start_time > deadline - 3:
                        return
                    if len(path) == N:
                        if path[0] in test_adj[path[-1]]:
                            h_found[0] = True
                            cycles[40] = list(path)
                        return
                    for nxt in test_adj[path[-1]]:
                        if nxt not in visited:
                            path.append(nxt)
                            visited.add(nxt)
                            hdfs(path, visited)
                            path.pop()
                            visited.remove(nxt)
                            if h_found[0]:
                                return
                hdfs([0], {0})
                if not h_found[0]:
                    continue
        
        # Check all other cycle lengths
        still_pancyclic = True
        for L in range(3, 40):
            c = cycles.get(L)
            if c is None:
                still_pancyclic = False
                break
            ok = True
            for i in range(L):
                u = c[i]
                v = c[(i + 1) % L]
                if v not in test_adj[u]:
                    ok = False
                    break
            if not ok:
                found = find_cycle_length_fast(L, test_adj)
                if not found:
                    still_pancyclic = False
                    break
        
        if still_pancyclic:
            all_edges = test_edges
            adj = test_adj
    
    # ---------- Final verification and repair ----------
    adj = build_adjacency(all_edges)
    for L in range(3, N + 1):
        c = cycles.get(L)
        if c is None:
            found = find_cycle_length_fast(L, adj)
            if found:
                cycles[L] = found
            else:
                # Last resort: try basic DFS
                found = find_cycle_length(L, adj, all_edges)
                if found and found != "timeout":
                    cycles[L] = found
    
    # Final check
    if not verify_cycles(all_edges, cycles):
        # Rebuild cycles from scratch if needed
        for L in range(3, N + 1):
            if L not in cycles or not verify_single_cycle(cycles.get(L), L, adj):
                found = find_cycle_length_fast(L, adj)
                if found:
                    cycles[L] = found
    
    edge_list = sorted(all_edges)
    cycles_out = []
    for L in range(3, N + 1):
        cycles_out.append(cycles[L])
    
    return {'edges': edge_list, 'cycles': cycles_out}

def verify_single_cycle(c, L, adj):
    """Verify a single cycle."""
    if c is None:
        return False
    if len(c) != L:
        return False
    if len(set(c)) != L:
        return False
    for i in range(L):
        u = c[i]
        v = c[(i + 1) % L]
        if v not in adj[u]:
            return False
    return True

# EVOLVE-BLOCK-END

# Call the construct function
construct()
