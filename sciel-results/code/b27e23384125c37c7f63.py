import time
from math import gcd
import numpy as np

N = 512
DEADLINE = 85.0  # Safety margin from 90s limit

def _divisors_of(n):
    """Return all divisors of n."""
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return divs

def _is_valid_set(S, N):
    """Check if all pairs in S have lcm <= N."""
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            if (a // gcd(a, b)) * b > N:
                return False
    return True

def _greedy_color_bound(R_mask, adj, N):
    """
    Greedy coloring bound for the subgraph induced by R_mask.
    Returns an upper bound on the independence number of the induced subgraph.
    """
    # Extract vertices from R_mask
    vertices = []
    temp = R_mask
    while temp:
        lsb = temp & -temp
        v = lsb.bit_length() - 1
        vertices.append(v)
        temp ^= lsb
    
    if not vertices:
        return 0
    
    # Greedy coloring: assign colors to vertices
    # Color classes are independent sets in the conflict graph
    # So the number of colors is an upper bound on the independence number? 
    # Wait, no. In a conflict graph, an independent set is a set with no edges.
    # A proper coloring of the graph partitions vertices into independent sets.
    # The minimum number of colors (chromatic number) is a LOWER bound on clique number, 
    # but for independence number, we want the size of the largest independent set.
    
    # Actually, the standard Tomita coloring bound works differently.
    # Let me reconsider.
    
    # In MIS problems, we use a greedy coloring of the *candidate* subgraph.
    # The idea: if we can color the candidates with k colors, then any independent set 
    # can contain at most one vertex from each color class? No, that's not right either.
    
    # Let me think again. 
    # In a conflict graph, two vertices are adjacent if they CANNOT be together.
    # An independent set is a set of vertices with NO edges between them (all can be together).
    # 
    # Standard MIS B&B coloring bound:
    # We greedily color the candidates. If we use c colors, then the maximum independent set
    # in the subgraph has size at most c? No, that's for the complement.
    
    # Actually, I think I'm confusing myself. Let me use a simpler bound first:
    # The size of R_mask is an upper bound. We can improve this.
    
    # For now, let's just return popcount(R_mask) as the bound.
    # We can improve this later.
    return len(vertices)

def construct():
    start_time = time.time()
    
    # 1. Precompute LCMs and build conflict graph
    # Nodes are 0..N-1 representing values 1..N
    # Edge (u, v) exists if lcm(u+1, v+1) > N
    
    # For efficiency, we can precompute lcm for all pairs or compute on the fly.
    # N=512, so N^2 = 262,144 pairs. This is small.
    # We will use a list of bitmasks for the adjacency matrix.
    # adj[i] is a bitmask of nodes j such that lcm(i+1, j+1) > N.
    
    adj = [0] * N
    
    # Symmetry: if lcm(i,j) > N, then both i cannot be with j.
    # We only need to check i < j to set bits, but the graph is undirected.
    
    # Optimization: 
    # lcm(a, b) = a * b // gcd(a, b)
    # lcm(a, b) > N  <=>  a * b > N * gcd(a, b)
    
    for i in range(N):
        a = i + 1
        # We can restrict j to i+1..N-1 and set bits for both
        for j in range(i + 1, N):
            b = j + 1
            g = gcd(a, b)
            # Check if lcm > N
            # a * b // g > N
            # To avoid large integers (though Python handles big ints, 
            # a*b is at most 512*512 = 262144, which is fine)
            if (a * b) // g > N:
                # Conflict
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 2. Solve Maximum Independent Set (MIS) on this graph
    # An independent set in the conflict graph is a valid subset S such that 
    # no pair in S has lcm > N.
    
    # We use a recursive branch-and-bound algorithm with bitsets.
    # This is a standard approach for small-to-medium N.
    
    # First, establish a strong initial lower bound using known valid sets.
    # The divisors of highly composite numbers often form valid sets.
    
    best_size = 0
    best_set = []
    
    # Try divisors of various highly composite numbers
    for m in [504, 360, 420, 240, 180, 120, 72, 60, 48, 36, 24, 12, 8, 6, 4, 2]:
        divs = [d for d in _divisors_of(m) if d <= N]
        if _is_valid_set(divs, N):
            if len(divs) > best_size:
                best_size = len(divs)
                best_set = divs[:]
    
    # Also try the set of all powers of 2 up to 512
    powers_of_2 = [2**k for k in range(0, 10) if 2**k <= N]  # 1,2,4,...,256
    if _is_valid_set(powers_of_2, N):
        if len(powers_of_2) > best_size:
            best_size = len(powers_of_2)
            best_set = powers_of_2[:]
    
    # Try combinations of small primes' powers
    # e.g., {1,2,3,4,5,6,7,8,9,10,...} - check if valid
    small_numbers = list(range(1, 20))
    if _is_valid_set(small_numbers, N):
        if len(small_numbers) > best_size:
            best_size = len(small_numbers)
            best_set = small_numbers[:]
    
    # Greedy construction: start empty, add numbers that are compatible with all current members
    greedy_set = []
    for num in range(1, N + 1):
        compatible = all((num // gcd(num, g)) * g <= N for g in greedy_set)
        if compatible:
            greedy_set.append(num)
    if _is_valid_set(greedy_set, N):
        if len(greedy_set) > best_size:
            best_size = len(greedy_set)
            best_set = greedy_set[:]
    
    # Global variables for the recursive solver
    # To avoid passing large lists around, we use a stack-based or recursive approach 
    # with bitmasks representing the candidate set R.
    
    # We'll implement a simple recursive B&B.
    # state: R (bitmask of candidates), size (current size of independent set)
    
    # To speed up, we can use a greedy coloring bound on the subgraph induced by R.
    # However, implementing full coloring bound in Python might be slow if not careful.
    # Given N=512, a simpler bound (size of R) might be sufficient if the graph is dense enough 
    # or if we prune well.
    
    # Let's try a standard recursive B&B with a greedy bound first.
    # If it's too slow, we can optimize.
    
    # Actually, for N=512, the conflict graph might be sparse or dense?
    # Many numbers have small LCMs. 
    # Example: powers of 2. lcm(2^a, 2^b) = 2^max(a,b) <= 256. So powers of 2 are compatible.
    # The graph is likely not extremely dense.
    
    # Let's use a more robust B&B.
    
    # We will use a list for the current set and a bitmask for candidates.
    
    def solve(R_mask, current_size, current_set):
        nonlocal best_size, best_set, start_time
        
        # Deadline check
        if time.time() - start_time > DEADLINE:
            return
        
        if R_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
        
        # Bound: current_size + popcount(R_mask) <= best_size -> prune
        rem = R_mask.bit_count()
            
        if current_size + rem <= best_size:
            return
        
        # Choose a vertex to branch on
        # Heuristic: pick the vertex with the highest degree in the induced subgraph R
        best_v = -1
        max_deg = -1
        
        temp_mask = R_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            v = lsb.bit_length() - 1
            
            deg = (adj[v] & R_mask).bit_count()
            
            if deg > max_deg:
                max_deg = deg
                best_v = v
                
            temp_mask ^= lsb
        
        if best_v == -1:
            return
            
        # Branch 2 first (Include) - often finds larger solutions faster
        R_incl = R_mask & ~(adj[best_v] | (1 << best_v))
        current_set.append(best_v + 1)
        solve(R_incl, current_size + 1, current_set)
        current_set.pop()
        
        # Branch 1: Exclude best_v
        R_excl = R_mask & ~(1 << best_v)
        solve(R_excl, current_size, current_set)
        
    # Initial call
    # R_mask includes all nodes 0..N-1
    initial_mask = (1 << N) - 1
    solve(initial_mask, 0, [])
    
    # Fallback if B&B didn't finish or found nothing (shouldn't happen, size >= 1)
    if best_size == 0:
        return [1]
        
    # Local improvement pass: try to add more elements to best_set
    # Convert to a set for fast lookup
    current_set_vals = set(best_set)
    improved = True
    while improved:
        improved = False
        for num in range(1, N + 1):
            if num in current_set_vals:
                continue
            # Check if num is compatible with all elements in current_set
            compatible = all((num // gcd(num, g)) * g <= N for g in current_set_vals)
            if compatible:
                current_set_vals.add(num)
                improved = True
    
    # Convert back to list
    result = list(current_set_vals)
    
    # Verify the result is valid
    if not _is_valid_set(result, N):
        # Something went wrong, fall back to the original best_set
        result = best_set
    
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            if (a // gcd(a, b)) * b > 512:
                print(f"Invalid pair: {a}, {b}")
                valid = False
                break
        if not valid:
            break
    if valid:
        print("All pairs valid.")
