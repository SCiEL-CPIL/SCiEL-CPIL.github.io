import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    n = 12
    m = 40
    max_mask = (1 << n) - 1
    
    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]
    
    def count_comparable(family):
        count = 0
        m = len(family)
        for i in range(m):
            mi = family[i]
            for j in range(i + 1, m):
                mj = family[j]
                if (mi & mj) == mi and mi != mj:
                    count += 1
                elif (mi & mj) == mj and mi != mj:
                    count += 1
        return count

    def get_delta(cand, family, index_to_remove):
        """
        Calculate the change in comparable pairs if we replace family[index_to_remove] with cand.
        Assumes cand is not in family.
        """
        old_mask = family[index_to_remove]
        delta = 0
        
        # Remove contributions of old_mask
        for i, mj in enumerate(family):
            if i == index_to_remove:
                continue
            # Check if old_mask and mj were comparable
            if (old_mask & mj) == old_mask and old_mask != mj:
                delta -= 1
            elif (old_mask & mj) == mj and old_mask != mj:
                delta -= 1
                
        # Add contributions of cand
        for i, mj in enumerate(family):
            if i == index_to_remove:
                continue
            # Check if cand and mj are comparable
            if (cand & mj) == cand and cand != mj:
                delta += 1
            elif (cand & mj) == mj and cand != mj:
                delta += 1
                
        return delta

    def initial_ball_B6():
        """
        Initialize with a 'Ball' (down-set) from the sub-lattice of vertices 0..5.
        This creates a very dense cluster of comparable sets.
        """
        # Generate all masks for vertices 0..5
        sub_masks = []
        for mask in range(1 << 6):
            sub_masks.append(mask)
            
        # Sort by popcount, then by value
        sub_masks.sort(key=lambda x: (bin(x).count('1'), x))
        
        # Take the first 40
        family = sub_masks[:m]
        return family

    def local_search(family, time_limit):
        """
        Hill Climbing Local Search.
        """
        current_score = count_comparable(family)
        best_score = current_score
        best_family = family[:]
        
        end_time = time.time() + time_limit
        used = set(family)
        
        improved = True
        while improved and time.time() < end_time:
            improved = False
            m = len(family)
            
            for i in range(m):
                if time.time() > end_time:
                    return best_family, best_score
                
                old_mask = family[i]
                
                # Try swapping with all other masks in the 12-vertex space
                # Optimization: We can limit the search space if we believe B6 is optimal,
                # but to be safe, we search all 4096 masks.
                # To speed up, we can pre-filter candidates that are likely better?
                # For now, iterate all. 40 * 4096 = 163,840 checks per pass.
                # Each check is O(m) = 40. Total ~6.5M ops per pass.
                # This is fast enough in Python if optimized, or we can sample.
                
                best_cand = None
                best_gain = 0
                
                for cand in range(1 << n):
                    if cand in used:
                        continue
                    
                    gain = get_delta(cand, family, i)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_cand = cand
                
                if best_cand is not None and best_gain > 0:
                    # Perform swap
                    used.remove(old_mask)
                    used.add(best_cand)
                    family[i] = best_cand
                    current_score += best_gain
                    improved = True
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_family = family[:]
                        
        return best_family, best_score

    # 1. Initialize
    family = initial_ball_B6()
    
    # 2. Local Search
    if time.time() < deadline - 5:
        time_remaining = deadline - 5 - time.time()
        family, score = local_search(family, time_remaining)
        
    # 3. Simulated Annealing Polish (Optional, if time permits)
    # Given the strength of the B6 ball, local search might be sufficient.
    # We can add a simple SA if time remains.
    if time.time() < deadline - 2:
        time_remaining = deadline - 2 - time.time()
        # Simple random restarts or SA
        # Let's stick to local search to avoid complexity and ensure stability.
        pass

    # Convert to list of lists
    result = []
    for mask in family:
        result.append(mask_to_list(mask))
    
    # Validate
    assert len(result) == m
    assert len(set(tuple(s) for s in result)) == m
    for s in result:
        assert isinstance(s, list)
        for x in s:
            assert isinstance(x, int)
            assert not isinstance(x, bool)
            assert 0 <= x <= 11
        assert len(s) == len(set(s))
        
    return result
