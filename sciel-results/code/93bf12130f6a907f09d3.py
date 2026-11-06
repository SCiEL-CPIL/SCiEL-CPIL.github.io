import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    N = 512

    # Precompute LCM table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            l = i * j // g
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # Build compatibility graph using bitmasks
    # Node i is compatible with node j if lcm(i, j) <= N
    # Represent each node's neighbors as a Python int bitmask
    # Nodes are indexed 1..N, but we use 0..N-1 for bit positions
    # Let's map node value v to index v-1 (0-indexed)
    
    compat = [0] * N  # compat[i] is a bitmask of neighbors of node i (0-indexed)
    
    for i in range(1, N + 1):
        idx = i - 1
        mask = 0
        for j in range(1, N + 1):
            if i != j and lcm_table[i][j] <= N:
                jidx = j - 1
                mask |= (1 << jidx)
        compat[idx] = mask

    # Maximum Clique algorithm using Bron-Kerbosch with pivot
    # We'll use a simple recursive approach with bitsets
    # For N=512, this might be slow if the graph is dense, but let's try.
    # Actually, 512 bits in a Python int is very efficient.
    
    best_clique = [0]
    best_size = 0
    
    def bron_kerbosch(R, P, X):
        nonlocal best_size, best_clique
        
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
            
        if not P:
            return
            
        # Pivot selection: choose u from P U X with max |P & N(u)|
        # Union of P and X
        U = P | X
        if U == 0:
            return
            
        # Find pivot u
        max_count = -1
        pivot = -1
        # Iterate through bits in U
        u_tmp = U
        while u_tmp:
            # Extract lowest set bit
            lsb = u_tmp & (-u_tmp)
            idx = lsb.bit_length() - 1
            u_tmp ^= lsb
            
            # Count neighbors in P
            count = (P & compat[idx]).bit_count()
            if count > max_count:
                max_count = count
                pivot = idx
            # Early termination: if count is 0, it's a good pivot
            if count == 0:
                break
        
        # If pivot is -1, something went wrong, just pick any
        if pivot == -1:
            lsb = U & (-U)
            pivot = lsb.bit_length() - 1

        # Candidates are P \ N(pivot)
        candidates = P & ~(compat[pivot])
        
        while candidates:
            # Pick a vertex v from candidates
            lsb = candidates & (-candidates)
            v = lsb.bit_length() - 1
            candidates ^= lsb
            
            R.append(v + 1)  # Store 1-indexed value
            
            # Recursive call: R ∪ {v}, P ∩ N(v), X ∩ N(v)
            P_new = P & compat[v]
            X_new = X & compat[v]
            
            bron_kerbosch(R, P_new, X_new)
            
            R.pop()
            
            # Move v from P to X
            P &= ~(1 << v)
            X |= (1 << v)
            
            # Time check
            if time.time() > deadline:
                return

    # Initialize
    initial_P = (1 << N) - 1  # All nodes are candidates
    initial_X = 0
    initial_R = []
    
    bron_kerbosch(initial_R, initial_P, initial_X)
    
    if best_clique:
        return best_clique
    
    # Fallback: Divisors of a highly composite number
    # 360 = 2^3 * 3^2 * 5. Divisors: (3+1)*(2+1)*(1+1) = 24.
    # Let's just return divisors of 360.
    divs = []
    for i in range(1, 361):
        if 360 % i == 0:
            divs.append(i)
    return divs

if __name__ == "__main__":
    print(construct())
