import numpy as np
import itertools
import time

def construct():
    D = 11
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    best_vectors = []
    
    # Try different norm squares
    for R2 in [2, 3, 4, 5, 6, 8, 10]:
        if time.time() > deadline:
            break
            
        # Generate all integer vectors with norm squared R2
        candidates = []
        # We can generate by iterating over possible values.
        # For small R2, values are small.
        # Max value is sqrt(R2).
        max_val = int(np.sqrt(R2)) + 1
        
        # Recursive generation or itertools.product
        # Range of each coord: -max_val to max_val
        # This is (2*max_val+1)^D which is huge for D=11.
        # Better: generate combinations of non-zero entries.
        
        # For R2=2: 2 non-zeros, values +/-1.
        # For R2=3: 3 non-zeros, values +/-1.
        # For R2=4: 
        #   4 non-zeros, values +/-1.
        #   1 non-zero, value +/-2.
        #   2 non-zeros, values +/-1, +/-1? No, 1^2+1^2=2.
        #   2 non-zeros, one is +/-2? 2^2+0=4. So (2,0..).
        #   Wait, (2, 0) has norm 4.
        #   (1, 1, 1, 1) norm 4.
        #   (2, 1, 1) norm 6.
        
        # General generation:
        # Find all integer solutions to sum(x_i^2) = R2.
        # This is a partition of R2 into squares.
        
        def gen_vectors(R2, D):
            vecs = []
            # Get all possible magnitude patterns
            # e.g. for R2=4: [4], [2,2], [1,1,1,1]
            # We need to find partitions of R2 into squares.
            
            # Simple recursion to find square partitions
            def find_partitions(n, max_val, current):
                if n == 0:
                    yield current
                    return
                for v in range(max_val, 0, -1):
                    sq = v*v
                    if sq > n: continue
                    yield from find_partitions(n-sq, v, current + [v])
            
            patterns = list(find_partitions(R2, int(np.sqrt(R2))+1, []))
            
            for p in patterns:
                # p is a list of magnitudes, e.g. [1, 1]
                # Number of non-zero elements is len(p)
                k_nonzero = len(p)
                if k_nonzero > D: continue
                
                # Choose positions for these non-zero elements
                for positions in itertools.combinations(range(D), k_nonzero):
                    # Assign signs
                    # The magnitudes are fixed in p, but we can permute them if they are distinct?
                    # The problem is symmetric, so we just need to consider unique sets of values.
                    # However, for greedy, we want all distinct vectors.
                    # We can generate all permutations of p?
                    # Or just generate all sign combinations for the fixed positions and fixed magnitudes?
                    # If p has distinct values, permutations matter.
                    # Let's generate all distinct vectors from this pattern.
                    
                    # To avoid duplicates, we can just generate all sign assignments for the sorted magnitudes?
                    # No, positions are fixed.
                    # Let's iterate over all permutations of p?
                    # If p = [1, 1], perms are just [1, 1].
                    # If p = [1, 2], perms are [1, 2], [2, 1].
                    
                    unique_perms = set(itertools.permutations(p))
                    for perm in unique_perms:
                        # perm[i] is magnitude for positions[i]
                        # Signs can be +/-
                        for signs in itertools.product([1, -1], repeat=k_nonzero):
                            v = [0] * D
                            for i, pos in enumerate(positions):
                                v[pos] = perm[i] * signs[i]
                            vecs.append(tuple(v))
            return vecs

        candidates = gen_vectors(R2, D)
        # Remove duplicates (though generation should be mostly unique, sets are safer)
        candidates = list(set(candidates))
        
        # Greedy packing
        current_set = []
        current_R2 = R2
        
        # Sort candidates? Maybe by some heuristic?
        # Random shuffle might help avoid local minima.
        # Let's try a few random shuffles.
        
        best_for_R2 = []
        for attempt in range(5): # 5 random restarts
            if time.time() > deadline:
                break
                
            np.random.shuffle(candidates)
            S = []
            for v in candidates:
                # Check if v can be added to S
                # Condition: for all u in S, ||v - u||^2 >= R2
                # Since ||v||^2 = R2 and ||u||^2 = R2, this is v.u <= R2/2
                valid = True
                for u in S:
                    dot = np.dot(v, u)
                    if dot > R2 // 2: # Integer check: dot <= 1.5 means dot <= 1
                         # Wait, R2/2 might not be integer.
                         # Condition: 2*R2 - 2*dot >= R2  => R2 >= 2*dot => dot <= R2/2
                         # Since dot is integer, dot <= floor(R2/2)
                         if 2 * dot > R2:
                             valid = False
                             break
                if valid:
                    S.append(v)
                    
            if len(S) > len(best_for_R2):
                best_for_R2 = S
                
        if len(best_for_R2) > len(best_vectors):
            best_vectors = best_for_R2
            
    return np.array(best_vectors, dtype=np.int64)
