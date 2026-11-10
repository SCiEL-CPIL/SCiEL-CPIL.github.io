"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations, combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    n = 256
    
    def is_sidon(s):
        """Check if set s is Sidon (all pair sums including a+a are unique)."""
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                s_val = s[i] + s[j]
                if s_val in sums:
                    return False
                sums.add(s_val)
        return True
    
    def get_diffs(s):
        """Get all nonzero differences from set s."""
        diffs = set()
        for i in range(len(s)):
            for j in range(len(s)):
                if i != j:
                    d = s[j] - s[i]
                    if d != 0:
                        diffs.add(d)
        return diffs
    
    def can_add_to_sidon(s, x, sum_set):
        """Check if x can be added to Sidon set s while maintaining Sidon property."""
        # Check x+x against existing sums
        xx = x + x
        if xx in sum_set:
            return False
        # Check x+a against existing sums
        for a in s:
            if (x + a) in sum_set:
                return False
        return True
    
    def get_sum_set(s):
        """Get all pair sums (including a+a) from set s."""
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        return sums
    
    def greedy_construct(order, max_size=64):
        """Greedy construction using given element order."""
        sets = [[], []]
        diff_sets = [set(), set()]
        sum_sets = [set(), set()]
        
        for x in order:
            if time.time() > deadline:
                break
            
            best_side = -1
            best_priority = None
            
            for side in (0, 1):
                if len(sets[side]) >= max_size:
                    continue
                
                # Check Sidon property
                if not can_add_to_sidon(sets[side], x, sum_sets[side]):
                    continue
                
                # Check disjoint differences
                new_diffs = set()
                for a in sets[side]:
                    new_diffs.add(x - a)
                    new_diffs.add(a - x)
                new_diffs.discard(0)
                
                if new_diffs & diff_sets[1 - side]:
                    continue
                
                # Priority: prefer the side with fewer elements to balance
                priority = (-len(sets[side]), x)
                
                if best_priority is None or priority < best_priority:
                    best_priority = priority
                    best_side = side
            
            if best_side >= 0:
                side = best_side
                sets[side].append(x)
                sum_sets[side].add(x + x)
                for a in sets[side]:
                    if a != x:
                        sum_sets[side].add(x + a)
                for a in sets[side]:
                    if a != x:
                        diff_sets[side].add(x - a)
                        diff_sets[side].add(a - x)
        
        return sets
    
    def local_improve(sets, order):
        """Try to add more elements by removing some and re-adding."""
        sets = [list(s) for s in sets]
        diff_sets = [get_diffs(s) for s in sets]
        sum_sets = [get_sum_set(s) for s in sets]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in order:
                if x in sets[0] or x in sets[1]:
                    continue
                for side in (0, 1):
                    if len(sets[side]) >= 64:
                        continue
                    if not can_add_to_sidon(sets[side], x, sum_sets[side]):
                        continue
                    new_diffs = set()
                    for a in sets[side]:
                        new_diffs.add(x - a)
                        new_diffs.add(a - x)
                    new_diffs.discard(0)
                    if new_diffs & diff_sets[1 - side]:
                        continue
                    sets[side].append(x)
                    sum_sets[side].add(x + x)
                    for a in sets[side]:
                        if a != x:
                            sum_sets[side].add(x + a)
                    diff_sets[side] |= new_diffs
                    improved = True
                    break
        return sets
    
    def spectral_guided_construct():
        """Use spectral clustering on difference graph to guide construction."""
        # Build adjacency matrix based on difference conflicts
        # Two elements conflict if their difference is "small" (more likely to cause conflicts)
        # Actually, we want to find elements that are "compatible"
        
        # For now, use a simple spectral approach: sort by "density" of potential conflicts
        # Elements with fewer conflicts with existing set members should be added first
        
        sets = [[], []]
        diff_sets = [set(), set()]
        sum_sets = [set(), set()]
        
        # Start with elements that have "nice" properties (e.g., powers of 2)
        # These tend to have unique differences
        initial_candidates = [1, 2, 4, 8, 16, 32, 64, 128]
        
        # Add initial candidates greedily
        for x in initial_candidates:
            if time.time() > deadline:
                break
            for side in (0, 1):
                if len(sets[side]) >= 64:
                    continue
                if not can_add_to_sidon(sets[side], x, sum_sets[side]):
                    continue
                new_diffs = set()
                for a in sets[side]:
                    new_diffs.add(x - a)
                    new_diffs.add(a - x)
                new_diffs.discard(0)
                if new_diffs & diff_sets[1 - side]:
                    continue
                sets[side].append(x)
                sum_sets[side].add(x + x)
                for a in sets[side]:
                    if a != x:
                        sum_sets[side].add(x + a)
                diff_sets[side] |= new_diffs
                break
        
        # Now greedily add remaining elements, prioritizing those with fewer conflicts
        remaining = [x for x in range(1, n + 1) if x not in sets[0] and x not in sets[1]]
        
        # Score each remaining element by how many conflicts it would have
        def conflict_score(x):
            score = 0
            for side in (0, 1):
                if len(sets[side]) >= 64:
                    continue
                # Count potential Sidon conflicts
                for a in sets[side]:
                    if (x + a) in sum_sets[side] or (x + x) in sum_sets[side]:
                        score += 1
                # Count difference conflicts
                for a in sets[side]:
                    d1 = x - a
                    d2 = a - x
                    if d1 in diff_sets[1 - side] or d2 in diff_sets[1 - side]:
                        score += 1
            return score
        
        # Sort by conflict score (ascending)
        remaining.sort(key=conflict_score)
        
        for x in remaining:
            if time.time() > deadline:
                break
            
            best_side = -1
            for side in (0, 1):
                if len(sets[side]) >= 64:
                    continue
                if not can_add_to_sidon(sets[side], x, sum_sets[side]):
                    continue
                new_diffs = set()
                for a in sets[side]:
                    new_diffs.add(x - a)
                    new_diffs.add(a - x)
                new_diffs.discard(0)
                if new_diffs & diff_sets[1 - side]:
                    continue
                best_side = side
                break
            
            if best_side >= 0:
                side = best_side
                sets[side].append(x)
                sum_sets[side].add(x + x)
                for a in sets[side]:
                    if a != x:
                        sum_sets[side].add(x + a)
                for a in sets[side]:
                    if a != x:
                        diff_sets[side].add(x - a)
                        diff_sets[side].add(a - x)
        
        return sets
    
    # Strategy: Try multiple approaches
    best_sets = [[], []]
    best_score = 0
    
    # Approach 1: Spectral-guided construction
    sets = spectral_guided_construct()
    sets = local_improve(sets, list(range(1, n + 1)))
    score = len(sets[0]) + len(sets[1])
    if score > best_score:
        best_score = score
        best_sets = [list(s) for s in sets]
    
    # Approach 2: Deterministic smaller-first ordering
    base_order = list(range(1, n + 1))
    sets = greedy_construct(base_order)
    sets = local_improve(sets, base_order)
    score = len(sets[0]) + len(sets[1])
    if score > best_score:
        best_score = score
        best_sets = [list(s) for s in sets]
    
    # Approach 3: Reverse ordering
    rev_order = list(range(n, 0, -1))
    sets = greedy_construct(rev_order)
    sets = local_improve(sets, rev_order)
    score = len(sets[0]) + len(sets[1])
    if score > best_score:
        best_score = score
        best_sets = [list(s) for s in sets]
    
    # Approach 4: Random orderings
    random.seed(42)
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        sets = greedy_construct(order)
        sets = local_improve(sets, order)
        score = len(sets[0]) + len(sets[1])
        if score > best_score:
            best_score = score
            best_sets = [list(s) for s in sets]
    
    # Final improvement pass on the best sets
    final_order = list(range(1, n + 1))
    best_sets = local_improve(best_sets, final_order)
    
    # Verify validity
    for s in best_sets:
        assert is_sidon(s), "Sidon property violated"
    
    d0 = get_diffs(best_sets[0])
    d1 = get_diffs(best_sets[1])
    assert not (d0 & d1), "Difference sets not disjoint"
    
    return best_sets

# EVOLVE-BLOCK-END
