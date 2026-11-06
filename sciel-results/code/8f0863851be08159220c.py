import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for output serialization etc.
    
    def get_seed_solution():
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
    
    def try_find_coloring(n, time_limit):
        """
        Attempts to find a 4-coloring of K_n with no monochromatic triangles
        using Simulated Annealing / Min-Conflicts Hybrid.
        """
        if n < 2:
            return [0] * (n * (n - 1) // 2)
        
        # Matrix representation: M[i][j] = color (symmetric, 0 on diagonal)
        M = np.zeros((n, n), dtype=np.int8)
        
        # Precompute indices for all edges (i, j) with i < j
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        edge_list = np.array(edges)
        
        # Initialize randomly
        for i, j in edges:
            M[i, j] = M[j, i] = random.randint(0, 3)
        
        def count_tris():
            total = 0
            # For each triangle (i, j, k) with i < j < k
            # Check if M[i,j] == M[j,k] == M[i,k]
            for i in range(n):
                Mi = M[i]
                for j in range(i + 1, n):
                    cij = Mi[j]
                    Mj = M[j]
                    # Only look at k > j
                    # Mask: Mj[j+1:] == cij and Mi[j+1:] == cij
                    # This is a bit expensive if done naively for every i,j.
                    # Vectorized approach:
                    # count += np.sum((Mj[j+1:] == cij) & (Mi[j+1:] == cij))
                    # This is O(n^3) but vectorized inner loop is fast.
                    if cij < 4: # always true
                        total += np.sum((Mj[j+1:] == cij) & (Mi[j+1:] == cij))
            return total
        
        # Initial count
        current_tris = count_tris()
        
        if current_tris == 0:
            # Flatten to lexicographic order
            res = []
            for i, j in edges:
                res.append(M[i, j])
            return res

        T = 10.0
        cooling = 0.999
        min_T = 0.01
        
        steps = 0
        max_steps = 50000
        
        while time.time() < time_limit and steps < max_steps:
            # Pick a random edge
            idx = random.randint(0, len(edges) - 1)
            i, j = edges[idx]
            old_c = M[i, j]
            
            # Calculate delta for changing (i, j) to new_c
            # Delta = (new_mono_tris_involving_ij) - (old_mono_tris_involving_ij)
            # A triangle (i, j, k) is mono if M[i,j] == M[j,k] == M[i,k].
            # Fixing i, j. For each k != i, j:
            # It contributes if M[j,k] == M[i,k].
            # Let's precompute counts for i, j.
            
            # count_old[c] = number of k such that M[i,k] == M[j,k] == c
            # If we change M[i,j] from old_c to new_c:
            # Old contribution to total_tris from triangles involving edge (i,j) is count_old[old_c].
            # New contribution is count_old[new_c].
            # Wait, this logic is slightly flawed because changing M[i,j] affects
            # the "monochromatic" status of triangles (i, j, k) ONLY if M[i,k] == M[j,k].
            # If M[i,k] != M[j,k], the triangle is never monochromatic regardless of M[i,j].
            # If M[i,k] == M[j,k] == c, then:
            #   If M[i,j] == c, it WAS mono.
            #   If M[i,j] != c, it is NOT mono.
            
            # So:
            # Old cost part = count of k where M[i,k]==M[j,k]==old_c
            # New cost part = count of k where M[i,k]==M[j,k]==new_c
            
            # Vectorized count:
            # mask_k = (M[i, :] == M[j, :])
            # counts = np.bincount(np.where(mask_k)[0] ... wait, we need the value c.
            # vals = M[i, :]
            # matches = M[i, :] == M[j, :]
            # counts = np.bincount(np.where(matches, vals, 4), minlength=5)[:4]
            
            matches = (M[i, :] == M[j, :])
            # Exclude i and j themselves?
            # M[i,i]=0, M[j,j]=0.
            # If k=i: M[i,i]=0, M[j,i]=old_c. If old_c==0, match.
            # But triangle (i,j,i) is not a triangle.
            # We must exclude k=i and k=j.
            # Matches array has length n.
            # We only care about k where k != i and k != j.
            # We can set matches[i] = False, matches[j] = False.
            
            # Optimization:
            # We don't actually need to build the full bincount if we just need counts for specific colors.
            # But bincount is fast in numpy.
            
            matches[i] = False
            matches[j] = False
            
            # Get the colors at matching positions
            # We want count of c such that M[i,k] == M[j,k] == c
            # This is equivalent to counting c in M[i, matches]
            counts = np.bincount(M[i, matches], minlength=4)
            
            # old_c is M[i,j]. Note M[i,j] is not in matches because we excluded j.
            # So counts[old_c] is the number of k (k!=i,j) such that M[i,k]==M[j,k]==old_c.
            # This is exactly the number of mono triangles involving (i,j) that are currently mono.
            
            best_delta = 0
            best_new_c = old_c
            
            for new_c in range(4):
                if new_c == old_c: continue
                # delta = counts[new_c] - counts[old_c]
                delta = counts[new_c] - counts[old_c]
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = new_c
            
            # Acceptance probability
            if best_delta <= 0:
                accept = True
            else:
                if T > min_T:
                    accept = random.random() < np.exp(-best_delta / T)
                else:
                    accept = False
            
            if accept and best_new_c != old_c:
                M[i, j] = best_new_c
                M[j, i] = best_new_c
                # Update current_tris
                # Old contribution was counts[old_c] (calculated above with old M[i,j]? No, counts depends on M[i,k], M[j,k] which didn't change)
                # Wait, the counts vector is based on M[i, k] and M[j, k] for k != i, j.
                # Changing M[i, j] does NOT change M[i, k] or M[j, k] for k != i, j.
                # So the counts vector is VALID for both old and new states regarding the OTHER edges.
                # The number of mono triangles involving edge (i, j) changes from counts[old_c] to counts[best_new_c].
                # So current_tris += (counts[best_new_c] - counts[old_c]) = best_delta.
                current_tris += best_delta
                
                if current_tris == 0:
                    res = []
                    for ii, jj in edges:
                        res.append(M[ii, jj])
                    return res
            else:
                # If not accepted, no change
                pass
            
            # Cool down
            T *= cooling
            steps += 1
            
            # Occasionally restart if stuck too long with no improvement?
            # Simple restarts handled by caller.
            
        return None

    # Try to find coloring for n=50 down to 33
    for n in range(50, 32, -1):
        time_left = deadline - time.time()
        if time_left < 2.0:
            break
        # Allocate time proportional to difficulty? 
        # Give more time to larger n.
        # n=50: 10s, n=49: 5s, ...
        t_limit = time.time() + min(time_left * 0.8, 8.0)
        
        # Multiple restarts for the same n if time permits?
        # The function above does one SA run.
        # Let's do a loop inside here for restarts.
        
        found = None
        restarts = 0
        max_restarts_for_n = 10
        while time.time() < t_limit and restarts < max_restarts_for_n:
            res = try_find_coloring(n, time.time() + 2.0) # 2s per restart
            if res:
                found = res
                break
            restarts += 1
        
        if found:
            return {'n': n, 'colors': found}
            
    # Fallback to seed
    return get_seed_solution()

if __name__ == "__main__":
    res = construct()
    print(res['n'])
    # print(res['colors'])
