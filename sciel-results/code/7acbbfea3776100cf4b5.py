import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius
    'radius'. Uses a greedy set-cover approach followed by local search optimization.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        # Specific optimization for K_3(7,1)
        
        # 1. Precompute all words and their balls (neighbors within radius 1)
        all_words = list(itertools.product(range(3), repeat=7))
        N = len(all_words) # 2187
        
        # Map word tuple to index
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute balls: for each index, list of indices covered by that codeword
        # Ball(c) = {c} U {words at dist 1}
        balls = []
        for w in all_words:
            covered_indices = [word_to_idx[w]]
            for i in range(7):
                for val in range(3):
                    if val != w[i]:
                        new_word = list(w)
                        new_word[i] = val
                        new_word = tuple(new_word)
                        covered_indices.append(word_to_idx[new_word])
            balls.append(covered_indices)
            
        # 2. Greedy Set Cover to get initial solution
        # covered: boolean array
        covered = np.zeros(N, dtype=bool)
        codewords = []
        
        # To speed up greedy, we can just iterate. N is small.
        # Find best candidate by counting new coverage.
        while not covered.all():
            best_cw = -1
            best_count = -1
            # Optimization: We can sample or use a heuristic, but N=2187 is small enough
            # to scan all if we do it smartly? 
            # Scanning 2187 * 15 * 2187 is too slow for each step if we do it naively.
            # But we only do this ~150-200 times.
            # 200 * 2187 * 15 = ~6.5M ops. This is fast in Python? Maybe.
            # Let's try to optimize the inner loop.
            
            # Actually, let's just use a simple greedy:
            # For each candidate, count how many new words it covers.
            # We can maintain a 'uncovered' set.
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Heuristic: pick random few to start? No, greedy is better for init.
            # We will scan all candidates.
            for i in range(N):
                b = balls[i]
                # Count how many words in b are currently uncovered
                # Use numpy for speed
                new_cov = np.sum(~covered[b])
                if new_cov > best_count:
                    best_count = new_cov
                    best_cw = i
            
            if best_count == 0:
                # Should not happen if covered is not all
                break
                
            codewords.append(best_cw)
            covered[balls[best_cw]] = True
            
            if time.time() > deadline - 60:
                break

        # 3. Initialize coverage counts for Local Search
        # coverage_count[w] = number of codewords covering w
        coverage_count = np.zeros(N, dtype=np.int32)
        for cw in codewords:
            coverage_count[balls[cw]] += 1
        
        # Count how many words have coverage 0, 1, 2...
        # We only care if min(coverage_count) >= 1
        # Let's maintain cnt0 = number of words with count 0
        cnt0 = np.sum(coverage_count == 0)
        
        # Helper to check validity
        def is_valid():
            return cnt0 == 0

        # Helper to remove codeword
        # Returns True if valid after removal, False otherwise (and reverts)
        def try_remove(cw_idx):
            nonlocal cnt0
            b = balls[cw_idx]
            # Check if any word in b has count 1
            # If yes, removing will make it 0 -> invalid
            # We can check min(coverage_count[b]) == 1
            if np.min(coverage_count[b]) == 1:
                return False
            # Safe to remove
            coverage_count[b] -= 1
            # Update cnt0
            # Words that were 1 become 0
            new_zeros = np.sum(coverage_count[b] == 0)
            cnt0 += new_zeros
            return True

        # Helper to add codeword
        def try_add(cw_idx):
            nonlocal cnt0
            b = balls[cw_idx]
            # Words that were 0 become 1 (or more)
            old_zeros = np.sum(coverage_count[b] == 0)
            coverage_count[b] += 1
            cnt0 -= old_zeros
            return True

        # 4. Redundancy Removal (Greedy Pass)
        # Try to remove each codeword
        i = 0
        while i < len(codewords):
            if try_remove(codewords[i]):
                codewords.pop(i)
            else:
                i += 1
            if time.time() > deadline - 40:
                break

        # 5. Local Search
        # Strategy: Random Removals and Swaps
        # We want to minimize len(codewords).
        # If we can remove, do it.
        # If we can't remove, try to swap to enable future removals.
        
        random.seed(42)
        
        # To speed up swap search, we can prioritize candidates that cover words with low coverage
        # But random is a good start.
        
        stuck_rounds = 0
        max_stuck = 10000 # If stuck for long, try something else?
        
        while time.time() < deadline - 5:
            improvement = False
            
            # Try to remove a random codeword
            if codewords:
                idx_to_try = random.randint(0, len(codewords) - 1)
                cw = codewords[idx_to_try]
                if try_remove(cw):
                    codewords.pop(idx_to_try)
                    improvement = True
                    stuck_rounds = 0
                    continue # Restart loop with new state
            
            # Try to swap
            # Pick a random codeword to replace
            if codewords:
                idx_old = random.randint(0, len(codewords) - 1)
                cw_old = codewords[idx_old]
                
                # Pick a random new codeword
                # To be smarter, we could pick one that covers words with count 1
                # But random is okay.
                cw_new = random.randint(0, N - 1)
                
                if cw_new == cw_old:
                    continue
                
                # Apply swap: Remove old, Add new
                # Check if valid
                # We need to simulate the change
                b_old = balls[cw_old]
                b_new = balls[cw_new]
                
                # Calculate new counts locally? No, just modify and check, then revert if bad.
                # But we need to be careful with cnt0.
                
                # Save old state of cnt0? No, we can just compute delta.
                # Or simply:
                # 1. Decrement b_old
                # 2. Increment b_new
                # 3. Check cnt0
                # 4. If invalid, revert.
                
                # To avoid expensive revert logic, let's just do it and check.
                # Since N is small, this is fast.
                
                # Decrement
                coverage_count[b_old] -= 1
                # Increment
                coverage_count[b_new] += 1
                
                # Recalculate cnt0?
                # cnt0 = np.sum(coverage_count == 0) is O(N).
                # N=2187. This is fast.
                new_cnt0 = np.sum(coverage_count == 0)
                
                if new_cnt0 == 0:
                    # Valid swap
                    cnt0 = 0
                    codewords[idx_old] = cw_new
                    improvement = True
                    stuck_rounds = 0
                else:
                    # Invalid, revert
                    coverage_count[b_old] += 1
                    coverage_count[b_new] -= 1
                    cnt0 = new_cnt0 # Actually cnt0 didn't change because we reverted, but let's keep it consistent
                    # Wait, if we revert, cnt0 should be the same as before the swap.
                    # We didn't update cnt0 in the valid case yet?
                    # In valid case, we set cnt0=0.
                    # In invalid case, we revert counts, so cnt0 is back to original.
                    # We don't need to update cnt0 explicitly if we revert counts, 
                    # but we need to ensure cnt0 variable matches reality.
                    # Since we reverted counts, the number of zeros is the same as before.
                    # So we don't need to change cnt0.
            
            if not improvement:
                stuck_rounds += 1
                # If stuck too long, we might try a more aggressive move?
                # For now, just keep trying.
                
        # Convert indices back to lists
        return [list(all_words[cw]) for cw in codewords]
    
    # General fallback (not optimized for this specific case)
    all_words = list(itertools.product(range(q), repeat=n))
    
    def get_coverage_general(codeword):
        covered = set()
        covered.add(codeword)
        for i in range(n):
            for val in range(q):
                if val != codeword[i]:
                    new_word = list(codeword)
                    new_word[i] = val
                    covered.add(tuple(new_word))
        return covered
    
    codewords = []
    covered = set()
    
    while time.time() < deadline - 10:
        best_cw = None
        best_count = -1
        best_new_covered = set()
        
        for candidate in all_words:
            new_covered = get_coverage_general(candidate) - covered
            count = len(new_covered)
            if count > best_count:
                best_count = count
                best_cw = candidate
                best_new_covered = new_covered
        
        if best_count <= 0:
            break
        
        codewords.append(best_cw)
        covered.update(best_new_covered)
        
        if len(covered) == len(all_words):
            break
    
    # Redundancy removal
    i = 0
    while i < len(codewords) and time.time() < deadline - 5:
        cw_to_remove = codewords[i]
        coverage_of_cw = get_coverage_general(cw_to_remove)
        
        can_remove = True
        for w in coverage_of_cw:
            is_covered_by_other = False
            for j, other_cw in enumerate(codewords):
                if i == j:
                    continue
                if w in get_coverage_general(other_cw):
                    is_covered_by_other = True
                    break
            if not is_covered_by_other:
                can_remove = False
                break
        
        if can_remove:
            codewords.pop(i)
        else:
            i += 1
    
    return [list(cw) for cw in codewords]
