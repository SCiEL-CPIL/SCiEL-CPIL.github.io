import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: The task says 150s limit, but we should aim for ~140s to be safe.
    # The prompt says "evaluator KILLS ... after 150s". 
    # We will use a strict deadline.
    DEADLINE = 145.0 

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_instance(values, time_budget):
        n = len(values)
        if n <= 2:
            return list(values)

        v_set = set(values)
        index_to_val = {v: i for i, v in enumerate(values)}
        
        # 1. Build Conflict Graph (Adjacency Matrix)
        # Edge (i, j) exists if v_i + v_j is in V (for i != j)
        # OR if 2*v_i == v_j (which implies v_i + v_i = v_j)
        # Note: The condition "a+b=c" with a,b,c in S.
        # If we select S, we must not have any i,j,k in S such that v_i+v_j=v_k.
        # This is equivalent to: S is an Independent Set in the graph G where
        # G has edge (i,j) if v_i + v_j in V (i != j) OR 2*v_i = v_j.
        # Wait, let's re-verify the equivalence.
        # If S is an Independent Set in G:
        # For any a, b in S:
        # If a != b, edge (a,b) exists iff a+b in V. If a+b in S, then a+b in V, so edge exists. Contradiction.
        # If a == b, edge (a, 2a) exists iff 2a in V. If a+a=2a in S, then 2a in V, so edge exists. Contradiction.
        # So yes, Max Sum-Free Subset == Max Independent Set in G.
        
        adj = np.zeros((n, n), dtype=np.int8)
        
        # Vectorized-ish construction
        # For each i, j > i:
        # Check if v_i + v_j in v_set
        # Check if 2*v_i == v_j
        # Check if 2*v_j == v_i
        
        # Since n is small (<=240), O(n^2) is fine.
        for i in range(n):
            vi = values[i]
            for j in range(i + 1, n):
                vj = values[j]
                if (vi + vj) in v_set:
                    adj[i, j] = 1
                    adj[j, i] = 1
                elif 2 * vi == vj:
                    adj[i, j] = 1
                    adj[j, i] = 1
                elif 2 * vj == vi:
                    adj[i, j] = 1
                    adj[j, i] = 1

        # Precompute degrees for greedy init
        degrees = np.sum(adj, axis=1)

        # 2. Greedy Initialization
        # Min-degree greedy is standard for MIS
        def greedy_init():
            rem = np.ones(n, dtype=bool)
            selected = np.zeros(n, dtype=bool)
            # We need to dynamically update degrees or just pick min degree among remaining
            # For n=240, O(n^2) per step is 240*240*240 ~ 13M ops, which is fast in Python if optimized, 
            # but let's use a simple loop.
            
            # To make it faster, we can just sort by degree once? No, degrees change.
            # But for a good start, simple min-degree greedy is okay.
            
            # Optimization: Use a list of remaining indices and filter
            rem_indices = list(range(n))
            # Calculate initial degrees
            cur_degrees = degrees.copy()
            
            while rem_indices:
                if time.time() - start_time > DEADLINE - 5:
                    break
                # Find index in rem_indices with min cur_degree
                min_idx = 0
                min_val = float('inf')
                for idx in rem_indices:
                    if cur_degrees[idx] < min_val:
                        min_val = cur_degrees[idx]
                        min_idx = idx
                
                selected[min_idx] = True
                # Remove min_idx and its neighbors from rem_indices
                new_rem = []
                for idx in rem_indices:
                    if idx == min_idx:
                        continue
                    if adj[min_idx, idx]:
                        # This neighbor is removed
                        # Update degrees of its other neighbors?
                        # For simplicity in greedy, we just mark it removed.
                        # Strict degree update is complex.
                        # Let's just remove it.
                        pass
                    else:
                        new_rem.append(idx)
                rem_indices = new_rem
                # Note: This greedy doesn't update degrees of remaining nodes correctly 
                # if we don't track which neighbors were removed. 
                # However, for a starting point, it's often good enough.
                # A better greedy: pick random, or pick min degree from *original* degrees?
                # Let's stick to this simple version for speed.
            
            return selected

        # 3. Simulated Annealing with 2-flip moves
        # State: x (binary vector of length n)
        # Energy: E = -sum(x) + lambda * sum(adj[i,j] * x[i] * x[j])
        # We want to minimize E.
        # Move: Pick two distinct indices i, j. Flip x[i] and x[j].
        # Why 2-flip? 
        # If we have a local optimum where adding any single vertex creates a conflict, 
        # but swapping one out and one in resolves it, 1-flip can't find it.
        
        # Initialize with greedy
        x = greedy_init()
        
        # Calculate initial energy
        # E = -np.sum(x) + lambda * (x.T @ adj @ x) / 2 
        # Wait, sum(adj[i,j] x_i x_j) counts each edge twice if symmetric.
        # Let P = x.T @ adj @ x. This is 2 * (number of edges in induced subgraph).
        # So conflicts = P / 2.
        
        LAMBDA = 10.0 # Penalty for conflicts
        
        def get_energy(x_vec):
            # Vectorized
            # Sum of x
            size = np.sum(x_vec)
            # Conflicts
            # x.T @ adj @ x
            conflicts = np.dot(x_vec, np.dot(adj, x_vec)) / 2.0
            return -size + LAMBDA * conflicts

        current_energy = get_energy(x)
        best_x = x.copy()
        best_energy = current_energy
        
        # If best is not valid (conflicts > 0), we need to fix it or accept penalty.
        # The score is based on SIZE of sum-free subset.
        # If we have conflicts, it's invalid.
        # So we must ensure final x has 0 conflicts.
        # We will track the best VALID solution separately.
        
        best_valid_x = np.zeros(n, dtype=bool)
        best_valid_size = 0
        
        # Check if initial is valid
        if np.dot(x, np.dot(adj, x)) == 0:
            best_valid_x = x.copy()
            best_valid_size = np.sum(x)
        
        # SA Parameters
        T = 5.0
        T_min = 0.01
        cooling = 0.999
        
        # Precompute row sums of adj for faster conflict delta calc?
        # Delta E for flipping i and j:
        # Let x' be x with i, j flipped.
        # This is complex to vectorize dynamically for arbitrary pairs.
        # Given n=240, we can just recalculate energy for the new state?
        # x.T @ adj @ x is O(n^2). 240^2 = 57600 ops.
        # In Python/NumPy, this is very fast (< 1ms).
        # 150s / 1ms = 150,000 iterations. That's a lot.
        
        # Let's optimize the move generation.
        # Instead of random i, j, we can try to find "good" pairs.
        # But random is safer for SA.
        
        # To speed up, we can use a list of active vertices (where x=1) and inactive (x=0).
        # But flipping changes membership.
        
        # Let's just run the loop.
        
        # Optimization: Use float32 for adj and x to speed up dot products?
        adj_f = adj.astype(np.float32)
        x_f = x.astype(np.float32)
        
        def get_energy_fast(x_f_vec):
            size = np.sum(x_f_vec)
            conflicts = np.dot(x_f_vec, np.dot(adj_f, x_f_vec)) / 2.0
            return -size + LAMBDA * conflicts
        
        current_energy = get_energy_fast(x_f)
        
        while T > T_min:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Generate a 2-flip move
            # Pick i, j randomly
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            # Apply flip
            x_f[i] = 1.0 - x_f[i]
            x_f[j] = 1.0 - x_f[j]
            
            new_energy = get_energy_fast(x_f)
            delta = new_energy - current_energy
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                current_energy = new_energy
                # Check if valid and better
                # Valid means conflicts == 0
                # conflicts = dot(x, adj @ x) / 2
                # We can check if current_energy == -size (since LAMBDA*0 = 0)
                # If LAMBDA is large enough, any conflict makes energy worse than -size.
                # But we want to track the best VALID one explicitly.
                if abs(current_energy + np.sum(x_f)) < 1e-5: # No conflicts
                    size = int(np.sum(x_f))
                    if size > best_valid_size:
                        best_valid_size = size
                        best_valid_x = x_f.copy()
            else:
                # Revert
                x_f[i] = 1.0 - x_f[i]
                x_f[j] = 1.0 - x_f[j]
            
            T *= cooling
            
            # If we have a very good solution, we might want to restart or cool faster?
            # For now, linear cooling.
            
        # Ensure we return a valid set
        # If best_valid_size is 0, we need a fallback
        if best_valid_size == 0:
             # Fallback: All odds
             res = [v for v in values if v % 2 != 0]
             return res
        
        # Convert best_valid_x back to values
        res = [values[i] for i in range(n) if best_valid_x[i] == 1.0]
        
        # Final verification
        if not is_sum_free(res):
             # This should not happen if our logic is correct, but just in case
             # Try to repair by removing vertices involved in conflicts
             s = set(res)
             changed = True
             while changed:
                 changed = False
                 for a in list(s):
                     for b in list(s):
                         if a + b in s:
                             # Remove one of them? Remove the one with most conflicts?
                             # Simple: remove a
                             s.remove(a)
                             changed = True
                             break
                     if changed:
                         break
             res = list(s)
        
        return res

    results = []
    # Instances: (60, 801), (120, 802), (240, 803)
    # Allocate time proportional to size? Or equal?
    # SA complexity is O(n^2 * steps). 
    # n=240 is 16x n=60. 
    # If we give equal time, n=240 will do fewer steps.
    # Let's give more time to larger instances.
    
    instances = [(60, 801), (120, 802), (240, 803)]
    
    # Weights: 1, 2, 4?
    weights = [1, 2, 4]
    total_weight = sum(weights)
    
    for i, (size, seed) in enumerate(instances):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 5:
             # Fallback for all remaining
             vals = instance(size, seed)
             res = [x for x in vals if x % 2 != 0]
             results.append(res)
             continue
        
        # Allocate budget
        budget = remaining * (weights[i] / total_weight)
        # Ensure minimum budget
        budget = max(budget, 2.0)
        
        vals = instance(size, seed)
        res = solve_instance(vals, budget)
        
        # Verify
        if not is_sum_free(res):
             res = [x for x in vals if x % 2 != 0]
        
        results.append(res)
        
    return results
