import time
import random
import sys

def _check_sidon_unique_sums(els):
    """Check that all pair sums a+b (a<=b) are unique."""
    sums = set()
    n = len(els)
    for i in range(n):
        for j in range(i, n):
            s = els[i] + els[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_nonzero_diffs(els):
    """Get set of nonzero differences (positive and negative)."""
    diffs = set()
    n = len(els)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = els[j] - els[i]
                if d != 0:
                    diffs.add(d)
    return diffs

def _is_valid_pair(A, B):
    """Check that A and B are Sidon and their nonzero difference sets are disjoint."""
    if not A or not B:
        return False
    if not _check_sidon_unique_sums(A):
        return False
    if not _check_sidon_unique_sums(B):
        return False
    diffsA = _get_nonzero_diffs(A)
    diffsB = _get_nonzero_diffs(B)
    if diffsA & diffsB:
        return False
    return True

class SidonBuilder:
    def __init__(self, n=256, max_size=64):
        self.n = n
        self.max_size = max_size
        
    def _can_add_to_sidon(self, x, current_set, used_sums):
        """Check if x can be added to current_set maintaining Sidon property."""
        # Check sums with existing elements
        for a in current_set:
            s = a + x
            if s in used_sums:
                return False
        # Check diagonal sum
        if 2 * x in used_sums:
            return False
        return True

    def _diffs_with(self, x, current_set):
        """Get differences between x and all elements in current_set."""
        return set(x - a for a in current_set) | set(a - x for a in current_set)

    def greedy_build(self, fixed_set, used_diffs_fixed, rng, order='ascending'):
        """
        Greedily build a set disjoint in differences from fixed_set.
        fixed_set: The other set (A if building B, or B if building A).
        used_diffs_fixed: Differences of the fixed set.
        """
        new_set = []
        used_sums = set()
        used_diffs_new = set()
        
        candidates = list(range(1, self.n + 1))
        if order == 'descending':
            candidates.sort(reverse=True)
        elif order == 'random':
            rng.shuffle(candidates)
        else:
            candidates.sort()
            
        for x in candidates:
            if len(new_set) >= self.max_size:
                break
            
            # 1. Check Sidon property internally
            if not self._can_add_to_sidon(x, new_set, used_sums):
                continue
                
            # 2. Check difference disjointness with fixed_set
            new_diffs = self._diffs_with(x, new_set)
            
            # Check against fixed set diffs
            if new_diffs & used_diffs_fixed:
                continue
                
            # Check against new_set's own diffs
            if new_diffs & used_diffs_new:
                continue
            
            # Add to set
            new_set.append(x)
            for a in new_set[:-1]:
                used_sums.add(a + x)
            used_sums.add(2 * x)
            used_diffs_new.update(new_diffs)
            
        return new_set

    def build_pair(self, rng, strategy='asc'):
        """
        Build a pair (A, B).
        """
        A = []
        B = []
        
        used_sums_A = set()
        used_sums_B = set()
        used_diffs_A = set()
        used_diffs_B = set()
        
        def try_add(x, target_list, used_sums, used_diffs_self, used_diffs_other):
            # Check Sidon sums
            for a in target_list:
                if a + x in used_sums:
                    return False
            if 2 * x in used_sums:
                return False
            
            # Check Diffs
            new_diffs = set()
            for a in target_list:
                d1 = x - a
                d2 = a - x
                new_diffs.add(d1)
                new_diffs.add(d2)
                
            # Must not conflict with other set's diffs
            if new_diffs & used_diffs_other:
                return False
                
            # Must not conflict with self's existing diffs
            if new_diffs & used_diffs_self:
                return False
                
            # Commit
            target_list.append(x)
            for a in target_list[:-1]:
                used_sums.add(a + x)
            used_sums.add(2 * x)
            used_diffs_self.update(new_diffs)
            return True

        # Phase 1: Build A
        candidates = list(range(1, self.n + 1))
        if strategy == 'desc':
            candidates.sort(reverse=True)
        elif strategy == 'rand':
            rng.shuffle(candidates)
        else:
            candidates.sort()
            
        for x in candidates:
            if len(A) >= self.max_size:
                break
            if try_add(x, A, used_sums_A, used_diffs_A, used_diffs_B):
                pass

        # Phase 2: Build B
        candidates_b = list(range(1, self.n + 1))
        if strategy == 'desc':
            candidates_b.sort(reverse=True)
        elif strategy == 'rand':
            rng.shuffle(candidates_b)
        else:
            candidates_b.sort()
            
        for x in candidates_b:
            if len(B) >= self.max_size:
                break
            if try_add(x, B, used_sums_B, used_diffs_B, used_diffs_A):
                pass

        # Phase 3: Try to extend A again with B's diffs now fixed
        candidates_a2 = list(range(1, self.n + 1))
        if strategy == 'desc':
            candidates_a2.sort(reverse=True)
        elif strategy == 'rand':
            rng.shuffle(candidates_a2)
        else:
            candidates_a2.sort()
            
        for x in candidates_a2:
            if len(A) >= self.max_size:
                break
            if x in A:
                continue
            if try_add(x, A, used_sums_A, used_diffs_A, used_diffs_B):
                pass

        # Phase 4: Try to extend B again
        candidates_b2 = list(range(1, self.n + 1))
        if strategy == 'desc':
            candidates_b2.sort(reverse=True)
        elif strategy == 'rand':
            rng.shuffle(candidates_b2)
        else:
            candidates_b2.sort()
            
        for x in candidates_b2:
            if len(B) >= self.max_size:
                break
            if x in B:
                continue
            if try_add(x, B, used_sums_B, used_diffs_B, used_diffs_A):
                pass

        return A, B

def _fast_local_search(A, B, n=256, max_iters=100, rng=None, deadline=None):
    """
    Faster local search using incremental state checks.
    """
    if rng is None:
        rng = random.Random()
    
    best_A = list(A)
    best_B = list(B)
    best_score = len(A) + len(B)
    
    # Pre-compute state for current best
    def compute_state(A, B):
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A.add(A[i] + A[j])
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B.add(B[i] + B[j])
                
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    diffs_A.add(A[j] - A[i])
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    diffs_B.add(B[j] - B[i])
                    
        return sums_A, sums_B, diffs_A, diffs_B

    sums_A, sums_B, diffs_A, diffs_B = compute_state(best_A, best_B)
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        improved = False
        
        # Try to add to A
        # We can try random candidates or scan all. Scanning all is O(n * k) which is 256*64 ~ 16k ops.
        # Doing this for both A and B is ~32k ops. Very fast.
        for x in range(1, n + 1):
            if x in best_A:
                continue
            # Check sums
            valid = True
            for a in best_A:
                if a + x in sums_A:
                    valid = False
                    break
            if valid and 2 * x in sums_A:
                valid = False
            if not valid:
                continue
                
            # Check diffs
            new_diffs = set()
            for a in best_A:
                new_diffs.add(x - a)
                new_diffs.add(a - x)
                
            if new_diffs & diffs_B:
                continue
            if new_diffs & diffs_A:
                continue
                
            # Valid addition
            best_A.append(x)
            for a in best_A[:-1]:
                sums_A.add(a + x)
            sums_A.add(2 * x)
            diffs_A.update(new_diffs)
            best_score += 1
            improved = True
            break # Restart scan to pick up new opportunities
            
        # Try to add to B
        if not improved:
            for x in range(1, n + 1):
                if x in best_B:
                    continue
                valid = True
                for b in best_B:
                    if b + x in sums_B:
                        valid = False
                        break
                if valid and 2 * x in sums_B:
                    valid = False
                if not valid:
                    continue
                    
                new_diffs = set()
                for b in best_B:
                    new_diffs.add(x - b)
                    new_diffs.add(b - x)
                    
                if new_diffs & diffs_A:
                    continue
                if new_diffs & diffs_B:
                    continue
                
                # Valid addition
                best_B.append(x)
                for b in best_B[:-1]:
                    sums_B.add(b + x)
                sums_B.add(2 * x)
                diffs_B.update(new_diffs)
                best_score += 1
                improved = True
                break

        if not improved:
            break

    return best_A, best_B

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 256
    max_size = 64
    
    best_A = [1]
    best_B = [2]
    best_score = 2
    
    rng = random.Random(42) # Fixed seed for reproducibility of the search path? 
    # Actually, random restarts benefit from different seeds. Let's use a master rng.
    master_rng = random.Random()
    
    builder = SidonBuilder(n, max_size)
    
    strategies = ['asc', 'desc', 'rand', 'rand', 'rand']
    
    # Run multiple restarts
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        strategy = strategies[restarts % len(strategies)]
        A, B = builder.build_pair(master_rng, strategy)
        
        if not A: A = [1]
        if not B: B = [2]
        
        if _is_valid_pair(A, B):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = list(A)
                best_B = list(B)
                
        # Fast local search
        if time.time() < deadline - 2:
            A_ls, B_ls = _fast_local_search(A, B, n, max_iters=50, rng=master_rng, deadline=deadline-2)
            if _is_valid_pair(A_ls, B_ls):
                score_ls = len(A_ls) + len(B_ls)
                if score_ls > best_score:
                    best_score = score_ls
                    best_A = list(A_ls)
                    best_B = list(B_ls)
                    
    # Final polish on the best found
    if time.time() < deadline - 1:
        A_final, B_final = _fast_local_search(best_A, best_B, n, max_iters=100, rng=master_rng, deadline=deadline-1)
        if _is_valid_pair(A_final, B_final):
            score_final = len(A_final) + len(B_final)
            if score_final > best_score:
                best_score = score_final
                best_A = list(A_final)
                best_B = list(B_final)

    if not _is_valid_pair(best_A, best_B):
        best_A = [1]
        best_B = [2]
        
    return [best_A, best_B]
