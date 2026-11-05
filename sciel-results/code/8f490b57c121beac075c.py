import time
import random
import numpy as np

def construct():
    """
    Construct a 4-colouring of K_n with no monochromatic triangles.
    Maximize n. Returns {'n': n, 'colors': [...]} where colors has one entry
    for each pair (i,j), i<j, in lexicographic order.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Helper to extract colors from matrices
    def matrices_to_colors(M, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(4):
                    if M[k][i, j]:
                        colors.append(k)
                        break
        return colors

    # Helper to run SA for a specific n
    def run_sa(n, time_limit):
        if time.time() > time_limit:
            return None
            
        # M[c] is a boolean matrix where M[c][i][j] is True if edge (i,j) has color c
        M = [np.zeros((n, n), dtype=bool) for _ in range(4)]
        
        # Initialize randomly
        # We need exactly one color per edge
        for i in range(n):
            for j in range(i + 1, n):
                c = random.randint(0, 3)
                M[c][i, j] = True
                M[c][j, i] = True
        
        # Calculate initial cost
        # Cost is number of monochromatic triangles
        # We can compute this incrementally, but for init, we do it once
        # Using trace(M_c^3) / 6 is O(n^3) but done once.
        # Or sum over k of (M_c[i,k] & M_c[j,k]) for all i,j where M_c[i,j] is true.
        
        def calc_cost():
            total = 0
            for c in range(4):
                Mc = M[c]
                # For each edge (i,j) in color c, count common neighbors in color c
                # This is sum(Mc @ Mc) / 2 ? No.
                # Number of triangles = sum_{i<j<k} Mc[i,j]Mc[j,k]Mc[k,i]
                # This is (1/6) * trace(Mc^3)
                # Or simply:
                # For each i, let N_i be neighbors of i in color c.
                # Count edges within N_i.
                # Efficient: 
                # Let A = Mc.
                # Triangles = (A @ A @ A).trace() / 6
                # But we can do:
                # For each i, count pairs (j,k) such that j,k in N_i and edge (j,k) exists.
                # This is sum_{i} ( (Mc[i,:] @ Mc).sum() ) / 2 ?
                # Let's use the property: 6 * Triangles = trace(A^3).
                # A^3 is expensive.
                # Alternative:
                # For each c:
                #   For each i:
                #     neigh = np.where(Mc[i,:])[0]
                #     # Count edges between nodes in neigh
                #     # This is O(n^2) per i, O(n^3) total.
                #     # For n=50, 50^3 = 125,000 ops. Very fast in NumPy.
                pass
            
            # Actually, let's just use the incremental logic for everything.
            # Initialize cost by summing contributions of all edges?
            # No, just calculate it once properly.
            total = 0
            for c in range(4):
                Mc = M[c]
                # Count triangles
                # For each i, for each j > i where Mc[i,j] is True:
                #   Count k > j where Mc[i,k] and Mc[j,k] are True.
                # This is O(n^3) but with low constant.
                # Or use matrix multiplication:
                # trace(Mc @ Mc @ Mc) / 6
                # Let's use the matrix multiplication for initialization.
                A2 = Mc @ Mc
                total += np.trace(A2 @ Mc)
            return total // 6

        current_cost = calc_cost()
        best_cost = current_cost
        best_M = [m.copy() for m in M]
        
        if best_cost == 0:
            return matrices_to_colors(best_M, n)

        T = 1.0
        T_min = 0.01
        cooling_rate = 0.9995
        
        steps = 0
        last_improve_time = time.time()
        
        while time.time() < time_limit:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            # Find old color
            old_k = -1
            for k in range(4):
                if M[k][i, j]:
                    old_k = k
                    break
            if old_k == -1:
                continue
            
            # Pick new color
            new_k = random.randint(0, 3)
            if new_k == old_k:
                continue
            
            # Calculate delta cost
            # Remove triangles of color old_k containing (i,j)
            # Add triangles of color new_k containing (i,j)
            
            # Triangles containing (i,j) with color c:
            # Count k != i,j such that M[c][i,k] and M[c][j,k] are True.
            # Note: M[c][i,j] is currently True for old_k, False for new_k.
            # When we flip, we remove the edge from old_k and add to new_k.
            
            # For old_k:
            # The edge (i,j) is present.
            # Common neighbors of i and j in color old_k:
            # common_old = np.sum(M[old_k][i, :] & M[old_k][j, :])
            # Note: This counts k=i and k=j as well?
            # M[old_k][i,i] is 0. M[old_k][j,j] is 0.
            # M[old_k][i,j] is 1. M[old_k][j,i] is 1.
            # So the sum includes the pair (i,j) themselves?
            # Let's check:
            # Vector v_i = M[old_k][i, :]
            # Vector v_j = M[old_k][j, :]
            # Dot product v_i . v_j
            # Term k=i: v_i[i]=0, so 0.
            # Term k=j: v_i[j]=1, v_j[j]=0, so 0.
            # Term k=k (k!=i,j): v_i[k]=M[old_k][i,k], v_j[k]=M[old_k][j,k].
            # So the dot product exactly counts the number of triangles of color old_k containing edge (i,j).
            
            common_old = np.dot(M[old_k][i, :], M[old_k][j, :])
            
            # For new_k:
            # The edge (i,j) is NOT present in M[new_k] yet.
            # We are adding it.
            # The number of triangles formed by adding (i,j) to color new_k is the number of common neighbors of i and j in color new_k.
            common_new = np.dot(M[new_k][i, :], M[new_k][j, :])
            
            delta_cost = common_new - common_old
            
            # Acceptance probability
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                # Accept
                # Update matrices
                M[old_k][i, j] = False
                M[old_k][j, i] = False
                M[new_k][i, j] = True
                M[new_k][j, i] = True
                
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_M = [m.copy() for m in M]
                    last_improve_time = time.time()
                    
                    if best_cost == 0:
                        return matrices_to_colors(best_M, n)
            else:
                # Reject (no state change)
                pass
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
            # Optional: Reheat if stuck
            if time.time() - last_improve_time > 5.0:
                T = 1.0
                last_improve_time = time.time()
                
        return None

    best_n = 32
    best_colors = None
    
    # Try to find n=50
    # We will try n from 50 down to 33
    # If we find a valid coloring for n, we return it.
    # Since 50 is known to exist, we prioritize it.
    
    # To maximize the chance, we run SA on n=50 for a significant portion of the time.
    # If it fails, we try n=49, etc.
    
    # However, random init for n=50 is very hard.
    # Let's try a few restarts for n=50.
    
    # Time allocation:
    # 80s for n=50 (multiple restarts)
    # 20s for n=49
    # 10s for n=48
    # ...
    
    # Actually, if n=50 is not found, lower n are easier.
    # But the score is n. So 50 is much better than 49.
    
    # Let's run SA for n=50 with multiple restarts.
    
    restarts = 0
    max_restarts = 10
    
    while restarts < max_restarts and time.time() < deadline - 20:
        t_limit = min(deadline - 20, time.time() + 10.0) # 10s per restart
        result = run_sa(50, t_limit)
        if result is not None:
            return {'n': 50, 'colors': result}
        restarts += 1
        
    # If 50 failed, try 49
    if time.time() < deadline - 10:
        t_limit = min(deadline - 10, time.time() + 10.0)
        result = run_sa(49, t_limit)
        if result is not None:
            return {'n': 49, 'colors': result}
            
    # If 49 failed, try 48
    if time.time() < deadline - 5:
        t_limit = min(deadline - 5, time.time() + 5.0)
        result = run_sa(48, t_limit)
        if result is not None:
            return {'n': 48, 'colors': result}
            
    # Fallback: Return the seed 32-vertex coloring
    # This ensures a valid score of 32 (which is > 0)
    def get_seed_coloring():
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}
        
    return get_seed_coloring()

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
