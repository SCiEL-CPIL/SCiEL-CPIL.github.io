import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 512
    nodes = list(range(1, n + 1))
    
    # 1. Build adjacency bitsets
    # adj[i] is an integer where bit j is set if gcd(i,j)*i/j <= 512 (i.e., lcm(i,j) <= 512)
    # We use 0-indexed bits for nodes 1..512, so bit (i-1) corresponds to node i.
    adj = [0] * (n + 1)
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            g = gcd(i, j)
            l = (i // g) * j
            if l <= n:
                # Set bit j-1 in adj[i] and bit i-1 in adj[j]
                adj[i] |= (1 << (j - 1))
                adj[j] |= (1 << (i - 1))
        
        if time.time() > deadline:
            # Fallback if graph construction is too slow (unlikely)
            return [1, 2]

    # 2. Generate initial best clique
    # A set of divisors of L is a valid clique.
    # Find L in [1, 512] with the most divisors.
    best_seed = []
    max_divs = 0
    for L in range(1, n + 1):
        divs = []
        for k in range(1, int(L**0.5) + 1):
            if L % k == 0:
                divs.append(k)
                if k != L // k:
                    divs.append(L // k)
        if len(divs) > max_divs:
            max_divs = len(divs)
            best_seed = divs
            
    # Also check if [1..22] is better? 22 divisors vs max divisors of 512 (which is 12? 512=2^9, divs=10. 360 has 24 divisors. 360<=512. 480 has 28 divisors? 480=2^5*3*5 -> 6*2*2=24. 420=2^2*3*5*7 -> 3*2*2*2=24. 360=2^3*3^2*5 -> 4*3*2=24. 480 has 24. 504=2^3*3^2*7 -> 4*3*2=24. 
    # Actually 480, 504, 420, 360, 336, 288, 240, 216, 180, 168, 144, 120, 96, 72, 60, 48, 40, 36, 30, 24, 20, 18, 16, 15, 14, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1.
    # Let's just trust the divisor count loop above.
    
    best_clique_size = len(best_seed)
    best_clique = best_seed
    
    # 3. Bron-Kerbosch with bitsets
    # R is a list of nodes in the current clique
    # P is a bitset of candidates
    # X is a bitset of excluded nodes
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique, best_clique_size
        
        if time.time() > deadline:
            return
            
        # Pruning
        if len(R) + P.bit_count() <= best_clique_size:
            return

        if P == 0:
            if X == 0:
                if len(R) > best_clique_size:
                    best_clique_size = len(R)
                    best_clique = list(R)
            return

        # Pivot selection
        # We want u in P | X that maximizes (P & adj[u]).bit_count()
        # To speed this up, we can just iterate through bits in P.
        # Since n=512, iterating 512 bits is fast.
        
        # Optimization: If P is small, just iterate P.
        # If P is large, we might sample? No, 512 is small.
        
        pivot = -1
        max_count = -1
        
        # Iterate bits in P
        p_temp = P
        while p_temp:
            # Get lowest set bit
            lsb = p_temp & -p_temp
            u = lsb.bit_length() # This gives index 1-based? 
            # If lsb is 1 (bit 0), bit_length is 1. Node is 1.
            # If lsb is 2 (bit 1), bit_length is 2. Node is 2.
            # So node u corresponds to index u in adj array.
            
            count = (P & adj[u]).bit_count()
            if count > max_count:
                max_count = count
                pivot = u
            p_temp ^= lsb
            
        if pivot == -1:
            return # Should not happen if P != 0

        # Candidates: P & ~adj[pivot]
        # Note: adj[pivot] does not have bit for pivot itself set in our construction?
        # We only added i!=j pairs. So adj[pivot] has 0 at bit pivot-1.
        # So P & ~adj[pivot] includes pivot if pivot is in P.
        # Standard BK: iterate v in P \ N(u).
        # If v is not neighbor of u, we process it.
        # If v is neighbor of u, it's excluded from this loop, but will be handled in recursion where u is in R?
        # No, the logic is:
        # For each v in P \ N(u):
        #   R.add(v)
        #   BK(R, P & N(v), X & N(v))
        #   R.remove(v)
        #   P.remove(v)
        #   X.add(v)
        
        candidates = P & ~adj[pivot]
        
        while candidates:
            lsb = candidates & -candidates
            v = lsb.bit_length()
            
            # Recurse
            new_P = P & adj[v]
            new_X = X & adj[v]
            
            R.append(v)
            bron_kerbosch(R, new_P, new_X)
            R.pop()
            
            # Update P and X
            P ^= lsb # Remove v from P
            X |= lsb # Add v to X
            
            candidates ^= lsb # Remove v from candidates
            
            if time.time() > deadline:
                return

    # Initial call
    # P = all nodes (bits 0 to 511 set)
    initial_P = (1 << n) - 1
    initial_X = 0
    initial_R = []
    
    bron_kerbosch(initial_R, initial_P, initial_X)
    
    return best_clique
