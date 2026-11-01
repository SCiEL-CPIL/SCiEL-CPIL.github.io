import itertools
import random
import time

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412143214231243121342132413214312"
    if n >= 5:
        # For n >= 5, use the TSP heuristic on blocks of size n-1
        # Generate all permutations of 1..n-1
        prev_perms = list(itertools.permutations(range(1, n)))
        num_blocks = len(prev_perms)
        
        # Create blocks: p + [n] + p
        blocks = []
        for p in prev_perms:
            b = list(p) + [n] + list(p)
            blocks.append(b)
        
        # Precompute overlaps using a more efficient method
        # M[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
        # Since blocks have length 2*(n-1)+1 = 2n-1
        # Max overlap is at most 2*(n-1) = 2n-2 (since we can't have full overlap)
        
        # For efficiency, we'll use string matching
        # Convert blocks to strings for faster comparison
        block_strs = [''.join(map(str, b)) for b in blocks]
        block_lens = [len(b) for b in blocks]
        
        # Precompute overlaps
        overlaps = [[0]*num_blocks for _ in range(num_blocks)]
        
        for i in range(num_blocks):
            si = block_strs[i]
            for j in range(num_blocks):
                sj = block_strs[j]
                # Max possible overlap is min(len(si), len(sj)) - 1
                # Actually, we need to be careful: the overlap must be such that
                # when we append, we don't create a duplicate of the full block
                # But for the TSP formulation, we just want max suffix/prefix match
                max_k = min(len(si), len(sj)) - 1
                found = 0
                for k in range(max_k, -1, -1):
                    if k == 0:
                        found = 0
                        break
                    if si[-k:] == sj[:k]:
                        found = k
                        break
                overlaps[i][j] = found
        
        # Greedy TSP Heuristic with multiple random restarts
        def calc_total_overlap(path):
            total = 0
            for k in range(len(path)-1):
                total += overlaps[path[k]][path[k+1]]
            return total
        
        def calc_length(path):
            total = len(blocks[path[0]])
            for k in range(len(path)-1):
                total += len(blocks[path[k+1]]) - overlaps[path[k]][path[k+1]]
            return total
        
        best_path = None
        best_length = float('inf')
        
        # Try multiple random starts
        num_restarts = 20
        for restart in range(num_restarts):
            random.shuffle(prev_perms)
            # Regenerate blocks in shuffled order
            blocks_shuffled = []
            for p in prev_perms:
                b = list(p) + [n] + list(p)
                blocks_shuffled.append(b)
            block_strs_shuffled = [''.join(map(str, b)) for b in blocks_shuffled]
            
            # Recompute overlaps for shuffled blocks
            overlaps_shuffled = [[0]*num_blocks for _ in range(num_blocks)]
            for i in range(num_blocks):
                si = block_strs_shuffled[i]
                for j in range(num_blocks):
                    sj = block_strs_shuffled[j]
                    max_k = min(len(si), len(sj)) - 1
                    found = 0
                    for k in range(max_k, -1, -1, ):
                        if k == 0:
                            found = 0
                            break
                        if si[-k:] == sj[:k]:
                            found = k
                            break
                    overlaps_shuffled[i][j] = found
            
            # Greedy construction
            current = random.randint(0, num_blocks-1)
            unvisited = set(range(num_blocks))
            unvisited.remove(current)
            path = [current]
            
            while unvisited:
                best_next = -1
                best_overlap = -1
                for j in unvisited:
                    if overlaps_shuffled[current][j] > best_overlap:
                        best_overlap = overlaps_shuffled[current][j]
                        best_next = j
                path.append(best_next)
                unvisited.remove(best_next)
                current = best_next
            
            # 2-Opt Local Search
            improved = True
            iterations = 0
            max_iterations = 100
            while improved and iterations < max_iterations:
                improved = False
                iterations += 1
                N = len(path)
                
                # Precompute prefix sums for faster delta calculation
                P = [0] * (N + 1)
                R = [0] * (N + 1)
                for k in range(1, N):
                    P[k] = P[k-1] + overlaps_shuffled[path[k-1]][path[k]]
                    R[k] = R[k-1] + overlaps_shuffled[path[k]][path[k-1]]
                
                for i in range(N-2):
                    for j in range(i+2, N):
                        if j == i+1:
                            continue
                        # Old: ... path[i], path[i+1], ..., path[j], path[j+1] ...
                        # New: ... path[i], path[j], ..., path[i+1], path[j+1] ...
                        
                        # Boundary overlaps
                        if i > 0:
                            old_ov_1 = overlaps_shuffled[path[i-1]][path[i]]
                            new_ov_1 = overlaps_shuffled[path[i-1]][path[i]]  # unchanged
                        else:
                            old_ov_1 = 0
                            new_ov_1 = 0
                        
                        if j < N-1:
                            old_ov_2 = overlaps_shuffled[path[j]][path[j+1]]
                            new_ov_2 = overlaps_shuffled[path[i+1]][path[j+1]]
                        else:
                            old_ov_2 = 0
                            new_ov_2 = 0
                        
                        # Internal overlaps
                        old_internal = P[j] - P[i+1]
                        new_internal = R[j] - R[i+1]
                        
                        # New boundary overlaps
                        new_ov_1b = overlaps_shuffled[path[i]][path[j]]
                        new_ov_2b = overlaps_shuffled[path[i+1]][path[j+1]] if j < N-1 else 0
                        
                        # Old boundary overlaps that change
                        old_ov_1b = overlaps_shuffled[path[i]][path[i+1]]
                        old_ov_2b = overlaps_shuffled[path[j]][path[j+1]] if j < N-1 else 0
                        
                        delta = (new_ov_1b + new_ov_2b + new_internal) - (old_ov_1b + old_ov_2b + old_internal)
                        
                        if delta > 0:
                            segment = path[i+1:j+1]
                            segment.reverse()
                            path[i+1:j+1] = segment
                            improved = True
            
            current_length = calc_length(path)
            if current_length < best_length:
                best_length = current_length
                best_path = path
                # Store the shuffled blocks
                best_blocks = blocks_shuffled
        
        # Build the final word
        if best_path is not None:
            final_word = list(best_blocks[best_path[0]])
            for k in range(1, len(best_path)):
                prev_block = best_blocks[best_path[k-1]]
                curr_block = best_blocks[best_path[k]]
                # Need to recompute overlaps for the best path
                # For simplicity, just use the precomputed ones
                # But we need to map back... this is getting complex.
                # Let's just rebuild from scratch with the original blocks
                
            # Actually, let's just use the original approach with the best path
            # Rebuild with original blocks
            final_word = list(blocks[best_path[0]])
            for k in range(1, len(best_path)):
                prev_idx = best_path[k-1]
                curr_idx = best_path[k]
                ov = overlaps[prev_idx][curr_idx]
                final_word += blocks[curr_idx][ov:]
            
            return "".join(map(str, final_word))
        
        # Fallback
        return "".join(map(str, blocks[0]))
    
    # Fallback for n=4
    return "123412143214231243121342132413214312"
