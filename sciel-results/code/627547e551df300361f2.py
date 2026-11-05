import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 58  # Safety margin before 60s
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Build adjacency/conflict sets using bitmasks for speed
    # adj[i] is a bitmask where bit j is set if i and j conflict
    adj = [0] * N
    
    # Identify vertices with self-loops (2*v is a square)
    # These vertices cannot be in the solution set.
    forbidden_mask = 0
    
    for i in range(N):
        v = i + 1
        # Check self-loop
        if 2 * v in squares:
            forbidden_mask |= (1 << i)
            continue # Skip building adj for forbidden? No, other nodes might conflict with it, 
                     # but since it's forbidden, we just won't select it. 
                     # However, for the graph logic, we can just ignore it in selection.
        
        # Find neighbors
        for sq in squares:
            u = sq - v
            if 1 <= u <= N:
                j = u - 1
                adj[i] |= (1 << j)
                # Since graph is undirected, we could set adj[j] here too, 
                # but the loop covers all i, so it will be set when i=j's turn comes.
                # Actually, to save time, we can just rely on the symmetry in the check:
                # if i conflicts with j, then when we check j, i will be in its adj list?
                # No, we build adj[i] for all i.
                # We must ensure adj is symmetric.
                # The loop `for sq in squares` finds all u such that v+u=sq.
                # So adj[i] contains j.
                # When we process j, we will find i.
                # So adj is symmetric.
    
    # Candidates are vertices without self-loops
    candidates = [i for i in range(N) if not (forbidden_mask & (1 << i))]
    
    # Precompute candidate masks for faster intersection
    # We will represent the current set S as a bitmask `S_mask`
    
    def is_valid_addition(v, S_mask):
        """Check if v can be added to S_mask."""
        # v must be a candidate
        if forbidden_mask & (1 << v):
            return False
        # v must not conflict with any node in S_mask
        # adj[v] contains bits for neighbors of v
        # If (adj[v] & S_mask) is 0, no conflict
        return (adj[v] & S_mask) == 0

    def greedy_solve(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        order = candidates[:]
        random.shuffle(order)
        
        S_mask = 0
        for v in order:
            if is_valid_addition(v, S_mask):
                S_mask |= (1 << v)
                
        return S_mask

    def modular_seed(modulus):
        """
        Generate a seed based on modular arithmetic.
        Try to find a set of residues R such that:
        1. For all r in R, 2r is not a square mod modulus.
        2. For all r1, r2 in R, r1+r2 is not a square mod modulus.
        Then select all numbers v in candidates such that v % modulus in R.
        """
        squares_mod = set()
        for k in range(1, int(math.isqrt(modulus**2)) + 1):
            squares_mod.add((k * k) % modulus)
        
        # Find safe residues
        safe_residues = []
        for r in range(modulus):
            if (2 * r) % modulus not in squares_mod:
                safe_residues.append(r)
        
        # Find max independent set of residues? 
        # Simple approach: Try all pairs of safe residues, and single residues.
        best_size = 0
        best_residues = None
        
        # Try single residues
        for r in safe_residues:
            # Count how many candidates have this residue
            count = 0
            for v in candidates:
                if (v + 1) % modulus == r: # v is 0-indexed, value is v+1
                    count += 1
            if count > best_size:
                best_size = count
                best_residues = [r]
        
        # Try pairs of residues
        for i in range(len(safe_residues)):
            for j in range(i, len(safe_residues)):
                r1 = safe_residues[i]
                r2 = safe_residues[j]
                # Check if r1+r2 is a square mod modulus
                if (r1 + r2) % modulus in squares_mod:
                    continue
                # Check if r1, r2 are safe (already checked)
                count = 0
                for v in candidates:
                    val = (v + 1) % modulus
                    if val == r1 or val == r2:
                        count += 1
                if count > best_size:
                    best_size = count
                    best_residues = [r1, r2]
        
        if best_residues is None:
            return 0
            
        S_mask = 0
        for v in candidates:
            val = (v + 1) % modulus
            if val in best_residues:
                S_mask |= (1 << v)
        
        return S_mask

    def local_improve(S_mask):
        """Try to add vertices that don't conflict."""
        improved = True
        while improved:
            improved = False
            for v in candidates:
                if (S_mask & (1 << v)):
                    continue
                if is_valid_addition(v, S_mask):
                    S_mask |= (1 << v)
                    improved = True
        return S_mask

    def local_search_1_1(S_mask):
        """
        1-1 Exchange Local Search.
        """
        improved = True
        while improved and (time.time() - start_time < DEADLINE - 2):
            improved = False
            # Try removing each v in S
            S_list = [i for i in range(N) if S_mask & (1 << i)]
            random.shuffle(S_list)
            
            for v in S_list:
                # Remove v
                S_temp = S_mask & ~(1 << v)
                
                # Try to add any w not in S_temp
                # We want to see if we can add a w that was blocked by v
                # Or just any w.
                # If we can add a w, and len(S_temp with w) > len(S_mask), we gain.
                # Since we removed 1, we need to add at least 1 to break even, 2 to improve.
                # 1-1 exchange implies removing 1 and adding 1. Net change 0.
                # This is usually not useful for *maximizing* unless it leads to a better state.
                # However, if we remove 1 and can add 1 that is "better" (e.g. allows future adds), 
                # it's hard to measure.
                # Standard 1-1 swap in Max Independent Set: Remove v, add w. 
                # This only helps if it unlocks something. 
                # Let's stick to 1-2 which has a clear +1 gain.
                
                # Check if we can add ANY w
                # If we can add a w, the size stays same.
                # We should probably skip pure 1-1 swaps unless we have time, 
                # as they don't increase size immediately.
                pass
        
        return S_mask

    def local_search_1_2(S_mask):
        """
        1-2 Exchange Local Search.
        Remove one vertex v, try to add two vertices u, w.
        Net gain: +1.
        """
        improved = True
        while improved and (time.time() - start_time < DEADLINE - 2):
            improved = False
            S_list = [i for i in range(N) if S_mask & (1 << i)]
            random.shuffle(S_list)
            
            for v in S_list:
                # Remove v
                S_temp = S_mask & ~(1 << v)
                
                # Find candidates that can be added to S_temp
                # A candidate u is addable if (adj[u] & S_temp) == 0
                available = []
                for u in candidates:
                    if (S_temp & (1 << u)):
                        continue
                    if (adj[u] & S_temp) == 0:
                        available.append(u)
                
                if len(available) < 2:
                    continue
                
                # Find an independent pair in available
                # We want u, w in available such that (adj[u] & (1 << w)) == 0
                found_pair = False
                for i in range(len(available)):
                    u = available[i]
                    for j in range(i + 1, len(available)):
                        w = available[j]
                        # Check if u and w conflict
                        if (adj[u] & (1 << w)) == 0:
                            # Apply move
                            S_mask = S_temp | (1 << u) | (1 << w)
                            improved = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                
                if found_pair:
                    break # Restart loop with new S_mask
        
        return S_mask

    best_size = 0
    best_mask = 0
    
    # Initial seeds
    seeds = []
    
    # Modular seeds
    for mod in [4, 8, 16, 32]:
        seeds.append(modular_seed(mod))
        
    # Random greedy seeds
    for _ in range(10):
        seeds.append(greedy_solve())
    
    # Iterate
    for seed in seeds:
        if time.time() - start_time > DEADLINE:
            break
            
        S_mask = local_improve(seed)
        
        if bin(S_mask).count('1') > best_size:
            best_size = bin(S_mask).count('1')
            best_mask = S_mask
            
        # Local search
        S_mask = local_search_1_2(S_mask)
        
        if bin(S_mask).count('1') > best_size:
            best_size = bin(S_mask).count('1')
            best_mask = S_mask

        # Random restarts if time permits
        while time.time() - start_time < DEADLINE - 5:
            S_new = greedy_solve()
            S_new = local_improve(S_new)
            S_new = local_search_1_2(S_new)
            
            if bin(S_new).count('1') > best_size:
                best_size = bin(S_new).count('1')
                best_mask = S_new
            else:
                # If no improvement in a few tries, break to try other seeds?
                # Or just keep going.
                pass

    # Convert bitmask back to list of 1-based values
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    return sorted(result)

# EVOLVE-BLOCK-END
