import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    
    # Step 1: Generate the standard recursive superpermutation to get the initial order of permutations
    # This gives us a valid starting point and the set of blocks.
    def get_standard_superperm(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return word

    # We need the order of (n-1) permutations used in the last step of the recursive construction for n.
    # The recursive construction for n uses the order of perms from S_{n-1}.
    # So we generate S_{n-1} and extract the order.
    
    if n == 2:
        return "121"
        
    # Get S_{n-1}
    word_prev = get_standard_superperm(n - 1)
    
    # Extract order of (n-1) permutations from word_prev
    # The windows of length n-1 in word_prev that are permutations.
    # Note: word_prev is a superpermutation of n-1 symbols.
    # The "blocks" in the step to n correspond to these permutations.
    perms_order = []
    seen = set()
    for i in range(len(word_prev) - (n - 1) + 1):
        window = tuple(word_prev[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms_order.append(window)
            
    # perms_order is a list of tuples, length (n-1)!
    # We want to reorder this list to maximize overlaps.
    # Overlap between p1 and p2 is longest suffix of p1 matching prefix of p2.
    
    def get_overlap(p1, p2):
        # p1, p2 are tuples of length n-1
        # Max overlap is n-2
        max_k = n - 2
        for k in range(max_k, -1, -1):
            if k == 0:
                return 0
            if p1[n-1-k:] == p2[:k]:
                return k
        return 0

    # Precompute overlap matrix? 
    # (n-1)! is 720 for n=7. 720*720 = 518,400 entries.
    # Each entry is a small int.
    # Computing each takes up to 5 comparisons.
    # Total ~2.5M operations. Fast.
    
    N = len(perms_order)
    overlap_matrix = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                overlap_matrix[i][j] = get_overlap(perms_order[i], perms_order[j])
            else:
                overlap_matrix[i][j] = 0 # Not used for transition to self, but keep 0
                
    # Current order is identity: 0, 1, ..., N-1
    current_order = list(range(N))
    
    def calc_length(order):
        # Length = N * (2n - 1) - sum(overlaps)
        # Overlaps are between order[i] and order[i+1]
        total_overlap = 0
        for i in range(N - 1):
            u = order[i]
            v = order[i+1]
            total_overlap += overlap_matrix[u][v]
        return N * (2 * n - 1) - total_overlap
    
    current_length = calc_length(current_order)
    
    # Local Search: 2-opt
    # Try swapping pairs? Or reversing segments?
    # Reversing a segment is a standard 2-opt move in TSP.
    # But here the graph is directed?
    # Overlap(p, q) != Overlap(q, p) generally.
    # So it's a directed TSP (Asymmetric TSP).
    # 2-opt for ATSP: reverse segment?
    # If we reverse a segment $v_i, \dots, v_j$, the edges inside are reversed.
    # Edge $v_{k} \to v_{k+1}$ becomes $v_{k+1} \to v_{k}$.
    # This changes the cost significantly.
    # Simpler: try swapping two elements $i, j$.
    # Or try inserting an element at a different position.
    
    # Let's try a simple hill climbing with random restarts or just random swaps.
    # Given 720 nodes, random swaps might be slow to find global optimum.
    # But we only need to beat 5913.
    # Standard length is 5913.
    # Max possible overlap per step is $n-2 = 5$.
    # Total steps 719.
    # Max total overlap $719 * 5 = 3595$.
    # Min length $720 * 13 - 3595 = 9360 - 3595 = 5765$.
    # We are at 5913.
    # Overlap in standard construction:
    # $5913 = 720 * 13 - \text{Overlap}_{std}$.
    # $\text{Overlap}_{std} = 9360 - 5913 = 3447$.
    # Average overlap $3447 / 719 \approx 4.79$.
    # Max is 5.
    # So we are already close to the theoretical max for this block structure.
    # The gap to 5905 is 8.
    # So we need to increase overlap by 8.
    
    # Let's try to optimize.
    start_time = time.time()
    timeout = 140 # Safety margin
    
    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        # Randomly select a pair of indices to swap
        # Or try all pairs? 720*720/2 = 259k pairs.
        # Cost to evaluate swap:
        # Removing i, j and reinserting.
        # Only edges (i-1, i), (i, i+1) and (j-1, j), (j, j+1) change.
        # So O(1) cost evaluation.
        
        # Let's iterate through all pairs in random order
        indices = list(range(N))
        random.shuffle(indices)
        
        for idx_i in indices:
            i = current_order[idx_i]
            for idx_j in indices:
                if idx_i == idx_j: continue
                j = current_order[idx_j]
                
                # Calculate delta cost
                # Current edges involving i and j
                # Note: i and j are values in the order list.
                # We need their positions.
                # This is slow if we use values.
                # Let's swap positions p and q in current_order.
                
        # This double loop with position lookup is messy.
        # Let's just do random swaps of positions.
        
        for _ in range(10000): # Fixed number of attempts per pass
            p = random.randint(0, N - 1)
            q = random.randint(0, N - 1)
            if p == q: continue
            
            # Swap current_order[p] and current_order[q]
            u = current_order[p]
            v = current_order[q]
            
            # Calculate old cost contribution
            # Edges: (p-1, p), (p, p+1), (q-1, q), (q, q+1)
            # Be careful with boundaries.
            
            def get_edge_cost(pos):
                # Cost of edge from pos to pos+1
                if pos < 0 or pos >= N - 1:
                    return 0
                a = current_order[pos]
                b = current_order[pos+1]
                return -overlap_matrix[a][b] # Minimize cost = Maximize overlap. 
                                             # Let's just track overlap.
                                             # Delta Overlap = New - Old.
                                             # We want Delta > 0.
            
            # This is getting complicated to calculate delta exactly without rebuilding.
            # Given N=720, rebuilding the sum is O(N) = 720 ops.
            # 10000 swaps * 720 ops = 7.2M ops.
            # This is fast in Python? Maybe 1-2 seconds.
            # We can do many passes.
            
            current_order[p], current_order[q] = v, u
            new_length = calc_length(current_order)
            if new_length < current_length:
                current_length = new_length
                improved = True
            else:
                current_order[p], current_order[q] = u, v
                
    # Construct the final word
    word = []
    for idx in range(N):
        p = perms_order[current_order[idx]]
        block = list(p) + [n] + list(p)
        if not word:
            word = block
        else:
            # Find overlap
            max_k = n - 2
            found = 0
            for k in range(max_k, -1, -1):
                if k == 0:
                    break
                if word[-k:] == block[:k]:
                    found = k
                    break
            word += block[found:]
            
    return "".join(map(str, word))
