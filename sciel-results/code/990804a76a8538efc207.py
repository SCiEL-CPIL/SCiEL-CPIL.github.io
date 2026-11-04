import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # We will maintain a list of vectors.
    # Condition: For all u, v in V, ||u - v||^2 >= max(||u||^2, ||v||^2).
    # Note: Since we are dealing with integer vectors, we can use squared distances to avoid sqrt.
    
    # Helper to check if a new vector v can be added to the set S
    def can_add(v, S):
        v_sq = np.dot(v, v)
        if v_sq == 0:
            return False
        for u in S:
            diff = u - v
            dist_sq = np.dot(diff, diff)
            u_sq = np.dot(u, u)
            # Condition: dist_sq >= max(v_sq, u_sq)
            if dist_sq < max(v_sq, u_sq):
                return False
        return True

    # Greedy initialization
    # Start with basis vectors scaled by 2. 
    # ||2e_i - 2e_j||^2 = 4+4=8. max(4,4)=4. 8>=4 OK.
    # ||2e_i - 0||^2 = 4. max(4,0)=4. 4>=4 OK. (But 0 is not allowed)
    # Let's just start with a small random set or empty and grow.
    
    # Better initialization: Use the known kissing configuration structure if possible?
    # In D=11, the kissing number is 504. 
    # A good starting point is the set of all vectors with coordinates in {0, 1, -1} 
    # that are pairwise compatible? No, that's too many to check all.
    
    # Let's try a greedy approach with random candidates.
    # To speed up, we can bucket vectors by their norm squared or just keep a list.
    # Since D=11 is small, list operations are fast.
    
    S = []
    
    # Add some "safe" vectors first to bootstrap.
    # Vectors of the form (2, 0, ..., 0) are mutually compatible with each other?
    # u=(2,0..), v=(0,2,0..). dist^2 = 4+4=8. max(4,4)=4. 8>=4. Yes.
    # So we can add all 22 vectors: +/- 2e_i.
    for i in range(D):
        v_pos = np.zeros(D, dtype=np.int8)
        v_pos[i] = 2
        S.append(v_pos)
        v_neg = np.zeros(D, dtype=np.int8)
        v_neg[i] = -2
        S.append(v_neg)
        
    # Now try to add more vectors.
    # Candidate generation:
    # 1. Vectors with small integer coordinates.
    # 2. We can use a pool of pre-generated candidates or generate on the fly.
    
    # Pre-generate a pool of random candidates to avoid overhead in the loop?
    # Or generate on the fly. Given 145s, we can generate many.
    
    # Let's use a random walk / local search on the set S.
    # Move: Remove one vector u from S, add a new vector v.
    # Accept if |S| increases or if |S| stays same but we are "stuck" and want to explore?
    # Actually, we just want to maximize |S|.
    
    # Greedy addition phase
    # Try adding random vectors.
    # We need to ensure we don't spend too much time checking can_add for vectors that are clearly bad.
    
    # Let's define a function to try to add a vector.
    # If it fails, maybe we can swap?
    
    # Strategy:
    # 1. Greedy add as many as possible from a random pool.
    # 2. Local search: Pick a random vector in S, remove it. Try to add 2 random vectors.
    #    If both are added, we gained 1 vector.
    #    If one is added, we gained 0, but maybe we are in a better position?
    #    Actually, simple hill climbing on size is good.
    
    rng = np.random.RandomState(42)
    
    def random_vector():
        # Generate a random integer vector with small entries
        # Most vectors in optimal configs have small norms.
        # Try coordinates in {-2, -1, 0, 1, 2}
        v = rng.randint(-2, 3, size=D)
        # Ensure it's not zero
        while np.all(v == 0):
            v = rng.randint(-2, 3, size=D)
        return v.astype(np.int8)
    
    # Phase 1: Greedy Addition
    # We will try to add vectors until we can't anymore or time runs out.
    # To make it efficient, we can maintain a list of "blocked" vectors?
    # But the constraint is global.
    
    # Let's just do a loop:
    # 1. Generate candidate v.
    # 2. Check can_add(v, S).
    # 3. If yes, add it.
    # 4. If no, discard.
    # Repeat until time limit or no progress for a while.
    
    # To improve success rate, we can prioritize vectors that are "orthogonal-ish" to existing ones?
    # Or just random.
    
    # Let's add a "pool" of candidates we've tried and failed, to avoid re-testing?
    # Or just generate new ones.
    
    # Optimization: 
    # If we have a large S, can_add is O(N). N can be up to 500.
    # 500 * 11 * 11 ops is small.
    # We can do ~100,000 checks in 1 second? 
    # 100,000 * 500 * 100 ops = 5e9 ops. Might be too slow in Python.
    # Let's check speed.
    
    # Better: Vectorize the check?
    # S is a list of arrays. 
    # We can stack S into a matrix M (N x D).
    # dist_sq = sum((M - v)**2, axis=1)
    # norms_sq = sum(M**2, axis=1)
    # condition: dist_sq >= max(v_sq, norms_sq)
    
    S_matrix = np.array(S, dtype=np.int16)
    S_norms_sq = np.sum(S_matrix**2, axis=1)
    
    def can_add_fast(v, S_matrix, S_norms_sq):
        v_sq = np.dot(v, v)
        if v_sq == 0:
            return False
        # M is (N, D), v is (D,)
        # diff = M - v
        # dist_sq = sum(diff**2, axis=1)
        # Using BLAS? 
        # dist_sq = S_norms_sq + v_sq - 2 * (S_matrix @ v)
        dots = S_matrix @ v
        dist_sq = S_norms_sq + v_sq - 2 * dots
        # Check condition
        # dist_sq >= max(v_sq, S_norms_sq)
        # This is equivalent to:
        # dist_sq >= v_sq  AND  dist_sq >= S_norms_sq
        # (S_norms_sq + v_sq - 2*dots) >= v_sq  => S_norms_sq >= 2*dots
        # (S_norms_sq + v_sq - 2*dots) >= S_norms_sq => v_sq >= 2*dots
        # So we need: S_norms_sq >= 2*dots  AND  v_sq >= 2*dots
        
        cond1 = S_norms_sq >= 2 * dots
        cond2 = v_sq >= 2 * dots
        
        return np.all(cond1 & cond2)

    # Update S_matrix and S_norms_sq when adding
    def add_vector(v):
        nonlocal S_matrix, S_norms_sq
        S_matrix = np.vstack([S_matrix, v.reshape(1, -1)])
        v_sq = np.dot(v, v)
        S_norms_sq = np.append(S_norms_sq, v_sq)

    # Phase 1: Greedy
    # We will iterate, trying to add random vectors.
    # If we fail to add many in a row, we might be stuck.
    # Then we switch to Phase 2: Local Search (Swaps).
    
    stuck_counter = 0
    max_stuck = 100
    
    while time.time() < deadline - 20.0:
        v = random_vector()
        if can_add_fast(v, S_matrix, S_norms_sq):
            add_vector(v)
            stuck_counter = 0
        else:
            stuck_counter += 1
            if stuck_counter > max_stuck:
                break
    
    # Phase 2: Local Search
    # Try to swap one vector in S with two new vectors.
    # Or one with one (if it allows adding more later? No, size stays same).
    # So we only accept swaps that increase size.
    # Move: Remove u from S. Try to add v1, v2.
    # If both can be added to S\{u}, then new size is |S| - 1 + 2 = |S| + 1.
    
    # To do this efficiently:
    # 1. Remove u. Update S_matrix, S_norms_sq.
    # 2. Try to add v1.
    # 3. If v1 added, try to add v2.
    # 4. If v2 added, accept.
    # 5. Else, revert.
    
    # We need a way to revert.
    # We can just copy the matrices? 
    # N is ~500. Copying 500x11 array is fast.
    
    # Let's implement a swap loop.
    
    # We need to be careful with indices.
    
    def remove_vector(idx):
        # Returns the vector removed
        v = S_matrix[idx]
        # Remove row
        S_matrix_new = np.delete(S_matrix, idx, axis=0)
        S_norms_sq_new = np.delete(S_norms_sq, idx)
        return v, S_matrix_new, S_norms_sq_new

    def try_add_two(S_m, S_ns):
        # Try to add two random vectors to the set represented by S_m, S_ns
        # Returns (success, new_S_m, new_S_ns, v1, v2)
        
        # Try a few pairs
        for _ in range(10): # Try 10 pairs
            v1 = random_vector()
            v2 = random_vector()
            
            if can_add_fast(v1, S_m, S_ns):
                # Add v1
                S_m1 = np.vstack([S_m, v1.reshape(1, -1)])
                ns1 = np.append(S_ns, np.dot(v1, v1))
                
                if can_add_fast(v2, S_m1, ns1):
                    # Add v2
                    S_m2 = np.vstack([S_m1, v2.reshape(1, -1)])
                    ns2 = np.append(ns1, np.dot(v2, v2))
                    return True, S_m2, ns2, v1, v2
                    
        return False, None, None, None, None

    # Main Loop for Local Search
    # We will pick a random vector to remove, and try to add two.
    # If successful, we update S.
    
    # To avoid getting stuck in local minima, we can also try removing 2 and adding 3?
    # That's more expensive. Let's stick to 1-for-2 first.
    
    # Also, we can try 1-for-1 if it leads to a state where we can add more?
    # That's complex. Let's just do size-increasing moves.
    
    # One issue: random_vector() might generate vectors that are never compatible.
    # We should bias towards vectors that are "similar" to existing ones?
    # Or just random.
    
    # Let's run the local search until time limit.
    
    # We need to handle the case where S is empty or very small.
    # But we started with 22 vectors.
    
    # Optimization: 
    # If we remove a vector, we should prioritize removing vectors with high norms?
    # High norms make it harder for others to be added.
    # So removing a high-norm vector might free up space for two low-norm vectors.
    
    # Let's sort indices by norm squared descending.
    # Pick from the top few.
    
    while time.time() < deadline - 5.0:
        # Pick a candidate to remove
        # Weights: higher norm -> higher probability of removal?
        # Or just random.
        # Let's try picking from the highest norms.
        
        # If N is small, just pick random.
        N = S_matrix.shape[0]
        if N == 0:
            break
            
        # Pick top 10% by norm
        if N > 10:
            top_k = max(1, N // 10)
            top_indices = np.argsort(S_norms_sq)[-top_k:]
            idx = top_indices[rng.randint(len(top_indices))]
        else:
            idx = rng.randint(N)
            
        v_removed, S_m_new, S_ns_new = remove_vector(idx)
        
        # Try to add two
        success, S_m_final, S_ns_final, v1, v2 = try_add_two(S_m_new, S_ns_new)
        
        if success:
            S_matrix = S_m_final
            S_norms_sq = S_ns_final
            # We gained 1 vector.
            # We can continue immediately.
        else:
            # Revert (do nothing, S_matrix unchanged)
            # To avoid wasting time on the same "bad" removal, we could blacklist?
            # But random selection will eventually pick different ones.
            pass
            
    # Final check: 
    # The vectors in S_matrix are the result.
    # Convert to list of lists for return.
    
    if S_matrix.size == 0:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    return S_matrix

