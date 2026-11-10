import time
import random
import itertools
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    if n <= 1:
        return "".join(str(i) for i in range(1, n + 1))
    
    # For n=7, we use a specialized approach
    # The standard recursive construction gives length 5913.
    # We optimize the ordering of permutations at each step to maximize overlap.
    
    word = [1]
    
    for m in range(2, n + 1):
        # Extract distinct (m-1) permutations from word
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        N = len(perms)
        
        # If N is small, we can do exact or simple greedy
        if N <= 8:
            ordered_perms = perms
        else:
            # Compute overlap matrix using NumPy
            # perms is a list of tuples of length m-1
            max_ov = m - 1
            perms_arr = np.array(perms, dtype=np.int8)
            
            # overlap_matrix[i,j] = max k such that perms[i][-k:] == perms[j][:k]
            overlap_matrix = np.zeros((N, N), dtype=np.int8)
            
            # We want to maximize the sum of overlaps in the path.
            # For each k from max_ov down to 1, find pairs with overlap >= k.
            # Since we want the MAX overlap, we can fill the matrix by checking k from max down to 1.
            # However, checking all k for all pairs is O(N^2 * max_ov).
            # N=720, max_ov=6. 720^2 * 6 = 3.1M operations.
            
            # Efficient vectorized approach:
            # For a fixed k, overlap[i,j] >= k iff perms[i][m-1-k:] == perms[j][:k]
            # We can iterate k from max_ov down to 1.
            # If overlap_matrix[i,j] is 0, and condition holds, set to k.
            
            for k in range(max_ov, 0, -1):
                # left: suffix of length k for all i
                # right: prefix of length k for all j
                left = perms_arr[:, m-1-k:]
                right = perms_arr[:, :k]
                
                # match[i,j] = 1 if left[i] == right[j]
                # Using broadcasting: left[:, None, :] == right[None, :, :]
                # Shape: (N, 1, k) == (1, N, k) -> (N, N, k)
                # Then np.all(..., axis=2) -> (N, N)
                
                # Memory check: N=720, k=6. (720, 720, 6) int8 is ~3MB. OK.
                match = np.all(left[:, None, :] == right[None, :, :], axis=2)
                
                # Only update where overlap_matrix is 0 (not yet set to a higher value)
                mask = match & (overlap_matrix == 0)
                overlap_matrix[mask] = k
            
            # Now we have the overlap matrix.
            # We want to find a Hamiltonian path maximizing sum(overlap_matrix[path[i]][path[i+1]])
            
            # Convert to list for faster access in SA? 
            # N=720. List of lists might be faster than numpy array indexing in Python loops for SA moves?
            # Actually, numpy array indexing ov[i,j] is fast.
            
            # Simulated Annealing
            # Initial state: Random permutation or Greedy
            # Let's start with a Greedy path to get a good initial solution.
            
            ov_list = overlap_matrix.tolist()
            
            def greedy_path():
                path = [0]
                visited = [False] * N
                visited[0] = True
                current = 0
                for _ in range(N - 1):
                    best_next = -1
                    best_ov = -1
                    # Find unvisited node with max overlap from current
                    for j in range(N):
                        if not visited[j]:
                            ov = ov_list[current][j]
                            if ov > best_ov:
                                best_ov = ov
                                best_next = j
                    visited[best_next] = True
                    path.append(best_next)
                    current = best_next
                return path

            def path_score(path):
                score = 0
                for i in range(N - 1):
                    score += ov_list[path[i]][path[i+1]]
                return score

            # Initial path
            current_path = greedy_path()
            current_score = path_score(current_path)
            
            # SA Parameters
            # Time budget for SA
            sa_start = time.time()
            sa_deadline = deadline - 10.0 # Leave 10s for construction
            
            T = 10.0 # Initial temperature
            T_min = 0.1
            alpha = 0.999 # Cooling rate
            
            # We will do a fixed number of iterations or until time runs out
            # Each iteration: pick a move (swap or reverse), calculate delta, accept/reject.
            
            # Precompute some things? 
            # Delta calculation for swap i, j (i < j):
            # Old edges: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
            # Actually, swapping positions p and q in the path:
            # Let path be P. Swap P[p] and P[q].
            # Affected edges:
            # If p=0: (P[0], P[1]) becomes (P[q], P[1])
            # If p=N-1: (P[N-2], P[N-1]) becomes (P[N-2], P[p])
            # Generally:
            # Edges (P[p-1], P[p]) and (P[p], P[p+1]) are removed.
            # Edges (P[q-1], P[q]) and (P[q], P[q+1]) are removed.
            # New edges:
            # (P[p-1], P[q]), (P[q], P[p+1]) -- wait, this is for adjacent swap.
            # For arbitrary swap:
            # The segment between p and q is reversed? No, just swap.
            # Let's stick to simple swaps for simplicity, or 2-opt (reverse segment).
            # 2-opt is better for TSP-like problems.
            
            def calculate_delta_swap(path, p, q):
                # p and q are indices in path
                if p == q: return 0
                if p > q: p, q = q, p
                
                # Indices in path
                # Nodes involved:
                # A = path[p-1] (if p>0)
                # B = path[p]
                # C = path[p+1] (if p<N-1)
                # D = path[q-1] (if q>0)
                # E = path[q]
                # F = path[q+1] (if q<N-1)
                
                # Old score contributions:
                old = 0
                if p > 0: old += ov_list[path[p-1]][path[p]]
                if p < N-1: old += ov_list[path[p]][path[p+1]]
                if q > 0: old += ov_list[path[q-1]][path[q]]
                if q < N-1: old += ov_list[path[q]][path[q+1]]
                
                # New score contributions:
                # path[p] becomes E, path[q] becomes B
                new = 0
                if p > 0: new += ov_list[path[p-1]][path[q]] # A -> E
                if p < N-1: new += ov_list[path[q]][path[p+1]] # E -> C
                if q > 0: new += ov_list[path[q-1]][path[p]] # D -> B
                if q < N-1: new += ov_list[path[p]][path[q+1]] # B -> F
                
                return new - old

            # SA Loop
            best_path = current_path[:]
            best_score = current_score
            
            iter_count = 0
            while time.time() < sa_deadline:
                # Pick random swap
                p = random.randint(0, N - 1)
                q = random.randint(0, N - 1)
                
                delta = calculate_delta_swap(current_path, p, q)
                
                # Acceptance
                if delta > 0 or random.random() < np.exp(delta / T):
                    # Accept
                    current_path[p], current_path[q] = current_path[q], current_path[p]
                    current_score += delta
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_path = current_path[:]
                
                # Cool down
                T *= alpha
                if T < T_min:
                    T = T_min
                    # Optional: Reheat?
                    # T = 5.0
                    # break? No, keep going with low T to polish.
                
                iter_count += 1
                # Check time more frequently
                if iter_count % 1000 == 0:
                    if time.time() > sa_deadline:
                        break
            
            ordered_perms = [perms[i] for i in best_path]
        
        # Build the new word
        new = []
        for idx in range(len(ordered_perms)):
            p = ordered_perms[idx]
            block = list(p) + [m] + list(p)
            
            if idx == 0:
                new += block
            else:
                # Overlap is determined by the previous permutation and current permutation
                # We already optimized the order to maximize these overlaps.
                # The overlap length is overlap_matrix[prev_idx][curr_idx]
                # But we need to know the indices.
                # In the loop above, we lost the indices if we just took perms[i].
                # We need to reconstruct the overlap or store it.
                
                # Let's re-calculate overlap for the final construction to be safe,
                # or better, store the path indices and use the matrix.
                
                # Actually, the construction logic in the original code recalculated overlap.
                # That was O(N * L) which is fine.
                # Let's just do that to avoid passing indices around.
                
                max_ov = min(len(new), len(block))
                ov = 0
                for k in range(max_ov, 0, -1):
                    if new[-k:] == block[:k]:
                        ov = k
                        break
                new += block[ov:]
        
        word = new
        
        if time.time() > deadline:
            break
    
    return "".join(map(str, word))
