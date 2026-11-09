"""
Stronger heuristic for covering designs C(v, k, t).
Strategy:
1. Start with a random initial covering (greedy or random blocks).
2. Use local search to improve the covering:
   - Try to replace a block with a better one (greedy block improvement).
   - Try to remove redundant blocks.
   - Add blocks if needed (shouldn't happen if starting with valid covering).
3. Time limit guard.
"""
import itertools
import time
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    all_points = set(range(v))
    all_t_subsets = set(itertools.combinations(range(v), t))
    
    # Greedy initial construction
    def greedy_construct():
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            # Pick the first uncovered subset
            target = min(uncovered)
            block = set(target)
            # Add points to maximize coverage
            while len(block) < k:
                best_point = None
                best_gain = -1
                for p in all_points - block:
                    # Count how many uncovered t-subsets this point would help cover
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_sub = tuple(sorted(rest + (p,)))
                        if new_sub in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                if best_point is not None and best_gain > 0:
                    block.add(best_point)
                else:
                    break
            if len(block) < k:
                # Fill randomly
                remaining = list(all_points - block)
                random.shuffle(remaining)
                for p in remaining:
                    if len(block) == k:
                        break
                    block.add(p)
            block = tuple(sorted(block))
            blocks.append(block)
            # Remove covered subsets
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
            if time.time() > deadline - 10:
                break
        return blocks
    
    # Start with greedy
    blocks = greedy_construct()
    blocks = list(set(blocks))  # Remove duplicates
    
    # Convert to list of sets for easier manipulation
    block_sets = [set(b) for b in blocks]
    
    # Function to check if covering is valid
    def is_valid(block_list):
        covered = set()
        for b in block_list:
            for sub in itertools.combinations(b, t):
                covered.add(sub)
        return covered == all_t_subsets
    
    # Local search improvements
    def try_improve(blocks_list, max_iterations=1000):
        blocks_list = list(blocks_list)
        n = len(blocks_list)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            if time.time() > deadline - 5:
                break
            
            # Try to remove redundant blocks
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                # Check if block i is redundant
                test_blocks = blocks_list[:i] + blocks_list[i+1:]
                covered = set()
                for b in test_blocks:
                    for sub in itertools.combinations(b, t):
                        covered.add(sub)
                if covered == all_t_subsets:
                    blocks_list.pop(i)
                    n -= 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try to replace a block with a better one
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                old_block = blocks_list[i]
                # Try adding one point
                for p in all_points - old_block:
                    new_block = old_block | {p}
                    if len(new_block) <= k:
                        # Check if this improves coverage
                        old_covered = set()
                        for b in blocks_list:
                            if b is old_block:
                                continue
                            for sub in itertools.combinations(b, t):
                                old_covered.add(sub)
                        
                        new_covered = set()
                        for b in blocks_list:
                            if b is old_block:
                                for sub in itertools.combinations(new_block, t):
                                    new_covered.add(sub)
                            else:
                                for sub in itertools.combinations(b, t):
                                    new_covered.add(sub)
                        
                        if new_covered == all_t_subsets:
                            blocks_list[i] = new_block
                            improved = True
                            break
                
                if improved:
                    break
        
        return blocks_list
    
    # Run local search
    block_sets = try_improve(block_sets, max_iterations=500)
    
    # Final check and return
    if time.time() > deadline:
        # Return what we have
        blocks_final = [tuple(sorted(b)) for b in block_sets]
        blocks_final = list(set(blocks_final))
        return blocks_final
    
    # Verify validity
    covered = set()
    for b in block_sets:
        for sub in itertools.combinations(b, t):
            covered.add(sub)
    
    if covered != all_t_subsets:
        # Fallback: use greedy
        blocks_final = greedy_construct()
        return [tuple(sorted(b)) for b in blocks_final]
    
    blocks_final = [tuple(sorted(b)) for b in block_sets]
    blocks_final = list(set(blocks_final))
    return blocks_final
