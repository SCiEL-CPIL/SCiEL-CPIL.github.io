import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    if n <= 6:
        # Standard construction is optimal for n<=6
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
                overlap = 0
                min_len = min(len(new), len(block))
                for o in range(min_len, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    if n == 7:
        # Generate all permutations of 1..6
        import itertools
        perms = list(itertools.permutations(range(1, 7)))
        N = len(perms) # 720
        
        # Precompute overlap matrix
        # overlap[i][j] = longest suffix of perms[i] matching prefix of perms[j]
        overlap_mat = [[0] * N for _ in range(N)]
        for i in range(N):
            pi = perms[i]
            for j in range(N):
                if i == j:
                    overlap_mat[i][j] = 6
                    continue
                pj = perms[j]
                ov = 0
                for k in range(5, -1, -1):
                    if pi[-k:] == pj[:k]:
                        ov = k
                        break
                overlap_mat[i][j] = ov

        def calc_overlap_sum(path):
            total = 0
            for i in range(len(path) - 1):
                total += overlap_mat[path[i]][path[i + 1]]
            return total

        def build_word(path):
            w = []
            for idx in path:
                perm = perms[idx]
                block = list(perm) + [7] + list(perm)
                overlap = 0
                min_len = min(len(w), len(block))
                for o in range(min_len, -1, -1):
                    if w[len(w) - o:] == block[:o]:
                        overlap = o
                        break
                w += block[overlap:]
            return "".join(map(str, w))

        # Initial path: Standard order (lexicographic)
        # The standard recursive construction produces a specific order.
        # Let's use lexicographic order as a baseline, then optimize.
        initial_path = list(range(N))
        
        best_path = initial_path[:]
        best_overlap = calc_overlap_sum(best_path)
        
        # Local Search: 2-Opt and Or-Opt
        def local_search(path, max_time):
            current_path = path[:]
            current_overlap = calc_overlap_sum(current_path)
            improved = True
            start_ls = time.time()
            
            while improved and (time.time() - start_ls) < max_time:
                improved = False
                
                # 2-Opt
                for i in range(N - 2):
                    if time.time() > deadline:
                        return current_path, current_overlap
                    for j in range(i + 1, N - 1):
                        # Check improvement
                        old_ov1 = overlap_mat[current_path[i]][current_path[i + 1]]
                        old_ov2 = overlap_mat[current_path[j]][current_path[j + 1]]
                        
                        new_ov1 = overlap_mat[current_path[i]][current_path[j]]
                        new_ov2 = overlap_mat[current_path[i + 1]][current_path[j + 1]]
                        
                        if new_ov1 + new_ov2 > old_ov1 + old_ov2:
                            # Reverse segment i+1 to j
                            current_path[i + 1:j + 1] = reversed(current_path[i + 1:j + 1])
                            current_overlap += (new_ov1 + new_ov2) - (old_ov1 + old_ov2)
                            improved = True
                            break
                    if improved:
                        break
                
                # Or-Opt (Move segment of length L=1,2,3)
                if not improved:
                    for L in [1, 2, 3]:
                        for i in range(N - L):
                            if time.time() > deadline:
                                return current_path, current_overlap
                            # Segment to move: current_path[i : i+L]
                            # Try inserting at position j
                            # To keep O(N^2), we check all j.
                            # Delta calculation:
                            # Remove seg.
                            # Insert at j.
                            # Neighbors change.
                            
                            # Case 1: Move forward (j > i+L-1)
                            # Case 2: Move backward (j < i)
                            
                            # Simplified: Just try all j and calculate delta if promising?
                            # Or just do a full delta check.
                            
                            # Let's try a simpler move: Swap adjacent (L=1, j=i+1 or i-1) is covered by 2-opt? No.
                            # Let's implement Or-Opt properly.
                            
                            seg = current_path[i:i+L]
                            # Remove seg
                            left_idx = i - 1
                            right_idx = i + L
                            
                            # Try inserting at every position k in the remaining list
                            # This is O(N) per i, so O(N^2) total.
                            for k in range(N - L):
                                if k == i: continue
                                
                                # Calculate new overlap
                                # This is complex to do in O(1) for arbitrary L.
                                # Given N=720, N^2 = 500k.
                                # If we do this for L=1,2,3, it's 1.5M checks.
                                # Each check is O(1) if we precompute?
                                # Actually, the neighbors of the segment change.
                                
                                # Let's skip complex Or-Opt for now and rely on 2-Opt + Random Restarts.
                                # 2-Opt on permutation TSP-like problems is very effective.
                                pass

            return current_path, current_overlap

        # Run multiple restarts
        best_word = build_word(best_path)
        
        for restart in range(20): # Max restarts
            if time.time() > deadline - 10:
                break
            
            # Random shuffle initialization
            current_path = list(range(N))
            random.shuffle(current_path)
            
            # Greedy NN initialization is often better than random
            # Let's try Greedy NN
            unvisited = set(range(N))
            current_path = []
            # Start with random
            start_node = random.choice(list(unvisited))
            unvisited.remove(start_node)
            current_path.append(start_node)
            
            while unvisited:
                last = current_path[-1]
                best_next = -1
                best_ov = -1
                for node in unvisited:
                    ov = overlap_mat[last][node]
                    if ov > best_ov:
                        best_ov = ov
                        best_next = node
                current_path.append(best_next)
                unvisited.remove(best_next)
                
            # Local Search
            current_path, current_overlap = local_search(current_path, 5.0)
            
            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_path = current_path[:]
                best_word = build_word(best_path)

        return best_word

    # Fallback for n > 7
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
            overlap = 0
            min_len = min(len(new), len(block))
            for o in range(min_len, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
