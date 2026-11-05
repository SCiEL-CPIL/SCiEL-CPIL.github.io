import itertools
import time
import numpy as np
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute for each block, which t-subsets it covers
    block_covers = []
    for i, block in enumerate(all_blocks):
        subs = list(itertools.combinations(block, t))
        block_covers.append([t_subset_to_idx[s] for s in subs])
    
    # Also precompute for each t-subset, which blocks cover it (inverse)
    subset_to_blocks = defaultdict(list)
    for b_idx, covered in enumerate(block_covers):
        for c in covered:
            subset_to_blocks[c].append(b_idx)
    
    def greedy_set_cover(weights=None, time_limit=10.0):
        """Greedy set cover with optional random weights for diversity."""
        if time.time() > deadline - time_limit:
            return []
        
        active = np.ones(num_t_subsets, dtype=np.float64)
        selected = []
        
        # Precompute gain for each block initially
        if weights is not None:
            gains = np.array([len(block_covers[i]) / (weights[i] + 1e-9) for i in range(num_blocks)], dtype=np.float64)
        else:
            gains = np.array([len(block_covers[i]) for i in range(num_blocks)], dtype=np.float64)
        
        while True:
            if time.time() > deadline - time_limit:
                break
            
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
            
            b_idx = int(np.argmax(gains))
            selected.append(b_idx)
            
            covered = block_covers[b_idx]
            for c in covered:
                if active[c] > 0:
                    active[c] = 0
                    # Update gains for all blocks that also cover this subset
                    for other_b in subset_to_blocks[c]:
                        if other_b != b_idx and gains[other_b] > 0:
                            gains[other_b] -= 1.0
            
            if np.all(active == 0):
                break
        
        return selected
    
    def remove_redundant(selected):
        """Remove blocks that don't cover any unique t-subset."""
        if not selected:
            return selected
        
        cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in selected:
            for c in block_covers[b_idx]:
                cover_counts[c] += 1
        
        changed = True
        while changed:
            changed = False
            to_remove = []
            for b_idx in selected:
                redundant = True
                for c in block_covers[b_idx]:
                    if cover_counts[c] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b_idx)
            
            if to_remove:
                for b_idx in to_remove:
                    for c in block_covers[b_idx]:
                        cover_counts[c] -= 1
                remove_set = set(to_remove)
                selected = [b for b in selected if b not in remove_set]
                changed = True
        
        return selected
    
    def two_exchange(selected, time_limit=5.0):
        """Try to improve by removing 2 blocks and adding 1-2 blocks to cover gaps."""
        if not selected or time.time() > deadline - time_limit:
            return selected
        
        current_size = len(selected)
        best_solution = list(selected)
        best_size = current_size
        
        cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in selected:
            for c in block_covers[b_idx]:
                cover_counts[c] += 1
        
        # Try removing pairs of blocks
        n = len(selected)
        for i in range(n):
            if time.time() > deadline - time_limit:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - time_limit:
                    break
                
                # Temporarily remove blocks i and j
                uncovered = []
                for c in block_covers[selected[i]]:
                    cover_counts[c] -= 1
                    if cover_counts[c] == 0:
                        uncovered.append(c)
                for c in block_covers[selected[j]]:
                    cover_counts[c] -= 1
                    if cover_counts[c] == 0:
                        uncovered.append(c)
                
                if not uncovered:
                    # Both blocks were redundant together
                    new_selected = [b for idx, b in enumerate(selected) if idx != i and idx != j]
                    if len(new_selected) < best_size:
                        best_solution = new_selected
                        best_size = len(new_selected)
                        selected = new_selected
                        # Recompute cover_counts
                        cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
                        for b_idx in selected:
                            for c in block_covers[b_idx]:
                                cover_counts[c] += 1
                        n = len(selected)
                        break
                else:
                    # Try to find one or two blocks that cover all uncovered subsets
                    uncovered_set = set(uncovered)
                    
                    # Try single block
                    found_single = False
                    for candidate in range(num_blocks):
                        if candidate in selected:
                            continue
                        covers_all = True
                        for c in uncovered_set:
                            if c not in block_covers[candidate]:
                                covers_all = False
                                break
                        if covers_all:
                            new_selected = [b for idx, b in enumerate(selected) if idx != i and idx != j]
                            new_selected.append(candidate)
                            if len(new_selected) < best_size:
                                best_solution = new_selected
                                best_size = len(new_selected)
                                selected = new_selected
                                cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
                                for b_idx in selected:
                                    for c in block_covers[b_idx]:
                                        cover_counts[c] += 1
                                n = len(selected)
                                found_single = True
                                break
                    
                    if not found_single:
                        # Try two blocks
                        # Find blocks that cover at least one uncovered subset
                        candidates = []
                        for candidate in range(num_blocks):
                            if candidate in selected:
                                continue
                            overlap = 0
                            for c in uncovered_set:
                                if c in block_covers[candidate]:
                                    overlap += 1
                            if overlap > 0:
                                candidates.append((overlap, candidate))
                        
                        candidates.sort(reverse=True)
                        candidates = candidates[:20]  # Limit to top 20
                        
                        for idx1 in range(len(candidates)):
                            if time.time() > deadline - time_limit:
                                break
                            c1 = candidates[idx1][1]
                            remaining = uncovered_set - set(block_covers[c1])
                            if not remaining:
                                new_selected = [b for idx, b in enumerate(selected) if idx != i and idx != j]
                                new_selected.append(c1)
                                if len(new_selected) < best_size:
                                    best_solution = new_selected
                                    best_size = len(new_selected)
                                    selected = new_selected
                                    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
                                    for b_idx in selected:
                                        for c in block_covers[b_idx]:
                                            cover_counts[c] += 1
                                    n = len(selected)
                                    break
                            else:
                                for idx2 in range(idx1 + 1, len(candidates)):
                                    c2 = candidates[idx2][1]
                                    if remaining.issubset(set(block_covers[c2])):
                                        new_selected = [b for idx, b in enumerate(selected) if idx != i and idx != j]
                                        new_selected.extend([c1, c2])
                                        if len(new_selected) < best_size:
                                            best_solution = new_selected
                                            best_size = len(new_selected)
                                            selected = new_selected
                                            cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
                                            for b_idx in selected:
                                                for c in block_covers[b_idx]:
                                                    cover_counts[c] += 1
                                            n = len(selected)
                                            break
                    # Restore cover_counts if we didn't break
                    for c in block_covers[selected[i]] if i < n else []:
                        pass  # This is complex, let's just recompute if needed
                    # Actually, we need to restore properly
                    # Let's recompute cover_counts from scratch for safety
                    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
                    for b_idx in selected:
                        for c in block_covers[b_idx]:
                            cover_counts[c] += 1
        
        return best_solution
    
    # Phase 1: Multi-start greedy
    best_solution = None
    best_size = float('inf')
    
    num_starts = 5
    start_budget = min(15.0, (deadline - time.time()) / num_starts)
    
    for s in range(num_starts):
        if time.time() > deadline - 30.0:
            break
        
        if s == 0:
            # First start: standard greedy (no weights)
            solution = greedy_set_cover(weights=None, time_limit=start_budget)
        else:
            # Subsequent starts: random weights for diversity
            weights = np.random.exponential(1.0, size=num_blocks)
            solution = greedy_set_cover(weights=weights, time_limit=start_budget)
        
        solution = remove_redundant(solution)
        
        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = list(solution)
    
    # Phase 2: 2-exchange local search on the best solution
    if best_solution is not None and time.time() < deadline - 15.0:
        best_solution = two_exchange(best_solution, time_limit=min(20.0, deadline - time.time() - 15.0))
        best_solution = remove_redundant(best_solution)
    
    # Phase 3: Try a few more random restarts if time permits
    if best_solution is not None and time.time() < deadline - 20.0:
        for _ in range(3):
            if time.time() > deadline - 20.0:
                break
            weights = np.random.exponential(1.0, size=num_blocks)
            solution = greedy_set_cover(weights=weights, time_limit=5.0)
            solution = remove_redundant(solution)
            if len(solution) < best_size:
                best_size = len(solution)
                best_solution = list(solution)
                # Try 2-exchange on the new best
                if time.time() < deadline - 15.0:
                    best_solution = two_exchange(best_solution, time_limit=5.0)
                    best_solution = remove_redundant(best_solution)
    
    # Final redundancy removal
    if best_solution is not None:
        best_solution = remove_redundant(best_solution)
    
    # Convert to output format
    blocks = [list(all_blocks[b_idx]) for b_idx in best_solution]
    return blocks
