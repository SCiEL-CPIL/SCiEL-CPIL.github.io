import time
import sys
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for safety
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Build the conflict graph using bitsets
    # Vertex v (1..256) maps to bit (v-1)
    # conflicts_mask[v] has bit j set if vertex (j+1) conflicts with vertex v
    conflicts_mask = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflicts_mask[i] |= (1 << (j - 1))
                conflicts_mask[j] |= (1 << (i - 1))
    
    # Build adjacency matrix for spectral analysis
    rows = []
    cols = []
    for i in range(1, N + 1):
        m = conflicts_mask[i]
        while m:
            bit = (m & -m).bit_length() - 1
            rows.append(i - 1)
            cols.append(bit)
            m &= m - 1
    
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    
    # Compute principal eigenvector (largest eigenvalue) of adjacency matrix
    # This identifies the "core" of the conflict structure
    # Vertices with high eigenvector values are in dense conflict regions
    # We want to AVOID these, so we use negative values as priority
    try:
        if len(rows) > 0:
            # Use eigsh for sparse matrix
            # We want the eigenvector corresponding to the largest eigenvalue
            k = min(5, N - 1)
            vals, vecs = eigsh(A, k=k, which='LA')
            # Take the eigenvector for the largest eigenvalue
            idx = np.argmax(vals)
            eigvec = vecs[:, idx]
            # Normalize to [0, 1]
            eigvec = (eigvec - eigvec.min()) / (eigvec.max() - eigvec.min() + 1e-10)
        else:
            eigvec = np.ones(N) / N
    except:
        eigvec = np.ones(N) / N
    
    # Priority: vertices with LOW eigenvector values are in sparse conflict regions
    # So we prioritize them for inclusion
    priority = 1.0 - eigvec  # Higher priority = less conflict
    
    def greedy_spectral(remaining_mask, rng):
        """
        Greedy MIS using spectral priority + randomization.
        remaining_mask: bitmask of available vertices (bit v-1 set means vertex v is available)
        Returns a list of vertices in the independent set.
        """
        solution = []
        while remaining_mask:
            # Get list of available vertices
            available = []
            m = remaining_mask
            while m:
                bit = (m & -m).bit_length() - 1
                available.append(bit + 1)
                m &= m - 1
            
            if not available:
                break
            
            # Score each available vertex: combine spectral priority and degree
            best_v = None
            best_score = float('-inf')
            candidates = []
            
            for v in available:
                deg = bin(conflicts_mask[v] & remaining_mask).count('1')
                # Combine spectral priority (higher is better) with low degree
                # Use a weighted combination
                score = priority[v - 1] * 10.0 - deg * 0.5
                if score > best_score + 1e-9:
                    best_score = score
                    candidates = [v]
                elif abs(score - best_score) <= 1e-9:
                    candidates.append(v)
            
            # Random choice among top candidates
            v = rng.choice(candidates)
            solution.append(v)
            # Remove v and its neighbors from remaining
            remaining_mask &= ~(1 << (v - 1))
            remaining_mask &= ~conflicts_mask[v]
        
        return solution
    
    def greedy_min_degree(remaining_mask, rng):
        """
        Randomized greedy MIS using min-degree heuristic.
        """
        solution = []
        while remaining_mask:
            available = []
            m = remaining_mask
            while m:
                bit = (m & -m).bit_length() - 1
                available.append(bit + 1)
                m &= m - 1
            
            if not available:
                break
            
            best_v = None
            best_deg = float('inf')
            candidates = []
            
            for v in available:
                deg = bin(conflicts_mask[v] & remaining_mask).count('1')
                if deg < best_deg:
                    best_deg = deg
                    candidates = [v]
                elif deg == best_deg:
                    candidates.append(v)
            
            v = rng.choice(candidates)
            solution.append(v)
            remaining_mask &= ~(1 << (v - 1))
            remaining_mask &= ~conflicts_mask[v]
        
        return solution
    
    def local_search_add(sol, remaining_mask):
        """
        Try to add vertices to the solution by checking if they can be added.
        Returns (improved_sol, new_remaining_mask)
        """
        sol_set = set(sol)
        # Compute conflicts with current solution
        sol_conflicts = 0
        for v in sol:
            sol_conflicts |= conflicts_mask[v]
        
        # Find vertices that can be added
        addable = []
        m = remaining_mask & ~sol_conflicts
        while m:
            bit = (m & -m).bit_length() - 1
            addable.append(bit + 1)
            m &= m - 1
        
        if not addable:
            return sol, remaining_mask
        
        # Sort by priority
        addable.sort(key=lambda v: -priority[v - 1])
        
        new_sol = list(sol)
        new_remaining = remaining_mask
        
        for v in addable:
            if time.time() > deadline - 2:
                break
            # Check if v can be added (no conflict with new_sol)
            can_add = True
            for w in new_sol:
                if w == v:
                    continue
                if conflicts_mask[v] & (1 << (w - 1)):
                    can_add = False
                    break
            if can_add:
                new_sol.append(v)
                new_remaining &= ~(1 << (v - 1))
                new_remaining &= ~conflicts_mask[v]
        
        return new_sol, new_remaining
    
    # Initial solutions
    remaining_mask = (1 << N) - 1
    
    # Spectral greedy
    best_sol = greedy_spectral(remaining_mask, random.Random(42))
    best_size = len(best_sol)
    
    # Min-degree greedy
    sol2 = greedy_min_degree(remaining_mask, random.Random(123))
    if len(sol2) > best_size:
        best_size = len(sol2)
        best_sol = sol2
    
    # Randomized restarts
    rng = random.Random()
    restart_count = 0
    while True:
        if time.time() > deadline - 3:
            break
        
        # Alternate between spectral and min-degree
        if restart_count % 2 == 0:
            sol = greedy_spectral((1 << N) - 1, rng)
        else:
            sol = greedy_min_degree((1 << N) - 1, rng)
        restart_count += 1
        
        if len(sol) > best_size:
            best_size = len(sol)
            best_sol = sol
        
        # Try local search to add more vertices
        if time.time() < deadline - 5:
            sol_set = set(best_sol)
            remaining_mask = (1 << N) - 1
            for v in best_sol:
                remaining_mask &= ~(1 << (v - 1))
                remaining_mask &= ~conflicts_mask[v]
            
            improved_sol, _ = local_search_add(best_sol, remaining_mask)
            if len(improved_sol) > best_size:
                best_size = len(improved_sol)
                best_sol = improved_sol
        
        # Try 1-swap local search occasionally
        if restart_count % 5 == 0 and time.time() < deadline - 8:
            # Try removing one vertex and adding two
            for idx in range(len(best_sol)):
                if time.time() > deadline - 4:
                    break
                v_remove = best_sol[idx]
                temp_sol = [v for i, v in enumerate(best_sol) if i != idx]
                temp_set = set(temp_sol)
                
                # Find vertices that can replace v_remove
                # First, compute conflicts with temp_sol
                temp_conflicts = 0
                for w in temp_sol:
                    temp_conflicts |= conflicts_mask[w]
                
                # Find candidates not in temp_set and not conflicting
                candidates = []
                for u in range(1, N + 1):
                    if u in temp_set or u == v_remove:
                        continue
                    if temp_conflicts & (1 << (u - 1)):
                        continue
                    candidates.append(u)
                
                if len(candidates) < 2:
                    continue
                
                # Sort by priority
                candidates.sort(key=lambda v: -priority[v - 1])
                
                # Try adding top 2 candidates
                added = 0
                new_sol = list(temp_sol)
                for u in candidates[:3]:
                    can_add = True
                    for w in new_sol:
                        if conflicts_mask[u] & (1 << (w - 1)):
                            can_add = False
                            break
                    if can_add:
                        new_sol.append(u)
                        added += 1
                        if added >= 2:
                            break
                
                if added >= 2 and len(new_sol) > best_size:
                    best_size = len(new_sol)
                    best_sol = new_sol
    
    return best_sol
# EVOLVE-BLOCK-END
