"""Program for large integer sets avoiding equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    """
    Return nonempty subset of [1,128] with no three distinct elements whose
    three pairwise greatest common divisors are equal. Maximize cardinality.
    
    Approach: Model as Maximum Independent Set in a 3-uniform hypergraph.
    Use Branch-and-Bound with bitset-based conflict propagation.
    """
    N = 128
    
    # 1. Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # 2. Precompute bad_thirds[u][v]
    # bad_thirds[u][v] is a bitset of all w such that {u, v, w} is a bad triple.
    # We assume u < v for indexing, but we can just make it a full (N+1)x(N+1) matrix for simplicity.
    bad_thirds = [[0] * (N + 1) for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_table[a][b]
            bitset = 0
            for c in range(b + 1, N + 1):
                if gcd_table[a][c] == gab and gcd_table[b][c] == gab:
                    bitset |= (1 << c)
            # Also check c < a and a < c < b? 
            # The triple is unordered. The condition is symmetric.
            # My loop above only checks c > b.
            # Let's just iterate all c != a, c != b.
            bitset = 0
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if gcd_table[a][c] == gab and gcd_table[b][c] == gab:
                    bitset |= (1 << c)
            bad_thirds[a][b] = bitset
            bad_thirds[b][a] = bitset
            
    # 3. Greedy Initialization
    # Try a few greedy orders to get a good lower bound for best_size
    def greedy_select(order):
        S = []
        # To speed up greedy, we can maintain a set of available vertices
        # But for N=128, O(N^3) is fine for initialization.
        for x in order:
            can_add = True
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    a = S[i]
                    b = S[j]
                    if gcd_table[a][b] == gcd_table[a][x] == gcd_table[b][x]:
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                S.append(x)
        return S

    # Try ascending, descending, and sorted by "conflict count"
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1))
    ]
    
    # Calculate conflict count for a heuristic order
    conflict_count = [0] * (N + 1)
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            # Count how many w form a bad triple with u and v
            # This is just popcount(bad_thirds[u][v])
            cnt = bad_thirds[u][v].bit_count()
            conflict_count[u] += cnt
            conflict_count[v] += cnt
            
    orders.append(sorted(range(1, N + 1), key=lambda x: conflict_count[x]))
    
    best_size = 0
    best_set = []
    
    for order in orders:
        sol = greedy_select(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = sol[:]
            
    # 4. Branch and Bound
    start_time = time.time()
    time_limit = 55.0  # Leave buffer for overhead
    
    # Global variables for the search
    global_best_size = best_size
    global_best_set = best_set[:]
    # To avoid copying lists too often, we pass the list and pop/append
    # But we need to store the best one.
    
    def solve(S_list, Available):
        nonlocal global_best_size, global_best_set
        
        # Time check
        if time.time() - start_time > time_limit:
            return
            
        current_size = len(S_list)
        avail_count = Available.bit_count()
        
        # Pruning: Upper bound
        if current_size + avail_count <= global_best_size:
            return
            
        if avail_count == 0:
            if current_size > global_best_size:
                global_best_size = current_size
                global_best_set = S_list[:]
            return
            
        # Heuristic: Choose vertex v from Available that maximizes removal
        # Removal = popcount(Available & new_bad)
        # where new_bad = OR(bad_thirds[u][v] for u in S_list)
        
        best_v = -1
        best_removal = -1
        best_new_bad = 0
        
        temp_avail = Available
        while temp_avail:
            lsb = temp_avail & -temp_avail
            v = lsb.bit_length() - 1
            temp_avail ^= lsb
            
            # Calculate new_bad for this v
            new_bad = 0
            for u in S_list:
                new_bad |= bad_thirds[u][v]
                
            # Count how many available vertices are removed
            removal = (Available & new_bad).bit_count()
            
            # Tie-breaker: Maybe prefer vertices with higher index? Or lower?
            # Just take the first one with max removal.
            if removal > best_removal:
                best_removal = removal
                best_v = v
                best_new_bad = new_bad
                
        if best_v == -1:
            return
            
        v = best_v
        new_bad = best_new_bad
        bit_v = 1 << v
        
        # Branch 1: Include v
        S_list.append(v)
        # New available: remove v, and remove any vertex that forms a bad triple with v and some u in S
        new_Available = Available & ~bit_v & ~new_bad
        solve(S_list, new_Available)
        S_list.pop()
        
        # Branch 2: Exclude v
        new_Available_skip = Available & ~bit_v
        solve(S_list, new_Available_skip)
        
    # Initial call
    initial_Available = 0
    for i in range(1, N + 1):
        initial_Available |= (1 << i)
        
    # Start with the greedy solution? 
    # If we start with S = best_set, we might miss a larger set that doesn't contain the greedy set.
    # However, starting with S=[] explores the whole space.
    # To speed up, we can start with the greedy set and try to add more?
    # No, standard MIS B&B starts with empty set.
    # But we have a good lower bound, so pruning will be effective early on.
    
    solve([], initial_Available)
    
    if not global_best_set:
        return [1]
        
    return sorted(global_best_set)

# EVOLVE-BLOCK-END
