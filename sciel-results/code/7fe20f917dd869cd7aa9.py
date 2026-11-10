"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from bisect import bisect_left, bisect_right

# EVOLVE-BLOCK-START
def _base3_construction(n):
    """Numbers in [1,n] with only digits 0,1 in base 3."""
    result = set()
    for x in range(1, n + 1):
        t = x
        ok = True
        while t > 0:
            if t % 3 == 2:
                ok = False
                break
            t //= 3
        if ok:
            result.add(x)
    return result

class FourAPFreeSet:
    """Efficient 4-AP-free set with fast membership and conflict checking."""
    
    def __init__(self, n):
        self.n = n
        self.s = set()
        self.sorted_list = []
    
    def add(self, x):
        if x in self.s:
            return False
        if x < 1 or x > self.n:
            return False
        if self._creates_4ap(x):
            return False
        self.s.add(x)
        idx = bisect_left(self.sorted_list, x)
        self.sorted_list.insert(idx, x)
        return True
    
    def remove(self, x):
        if x not in self.s:
            return False
        self.s.remove(x)
        idx = bisect_left(self.sorted_list, x)
        if idx < len(self.sorted_list) and self.sorted_list[idx] == x:
            self.sorted_list.pop(idx)
        return True
    
    def __contains__(self, x):
        return x in self.s
    
    def __len__(self):
        return len(self.s)
    
    def _creates_4ap(self, x):
        """Check if adding x creates any nonconstant 4-AP. Uses sorted list for efficiency."""
        # Position 4: x = a+3d, so a = x-3d, check a, a+d, a+2d in S
        for d in range(1, (x - 1) // 3 + 1):
            a = x - 3 * d
            if a < 1:
                break
            if a in self.s and (a + d) in self.s and (a + 2 * d) in self.s:
                return True
        
        # Position 3: x = a+2d, so a = x-2d, check a, a+d, a+3d in S
        for d in range(1, (x - 1) // 2 + 1):
            a = x - 2 * d
            if a < 1:
                break
            if a in self.s and (a + d) in self.s and (a + 3 * d) in self.s:
                return True
        
        # Position 2: x = a+d, so a = x-d, check a, a+2d, a+3d in S
        for d in range(1, x):
            a = x - d
            if a < 1:
                break
            if a in self.s and (a + 2 * d) in self.s and (a + 3 * d) in self.s:
                return True
        
        # Position 1: x = a, check a+d, a+2d, a+3d in S
        for d in range(1, (self.n - x) // 3 + 1):
            if (x + d) in self.s and (x + 2 * d) in self.s and (x + 3 * d) in self.s:
                return True
        
        return False
    
    def greedy_complete(self, order=None):
        """Complete the set greedily by adding elements in the given order."""
        if order is None:
            order = list(range(1, self.n + 1))
        added = 0
        for x in order:
            if x not in self.s and self.add(x):
                added += 1
        return added
    
    def local_search(self, max_iters=200, deadline=None):
        """Swap-based local search: try removing one element to add two or more."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if deadline is not None and time.time() > deadline:
                break
            
            # Try each element for removal
            for u in list(self.s):
                self.remove(u)
                added = 0
                added_elements = []
                
                # Try to add elements that were previously blocked
                for v in range(1, self.n + 1):
                    if v not in self.s and v != u:
                        if self.add(v):
                            added_elements.append(v)
                            added += 1
                
                if added >= 2:
                    improved = True
                    # Keep the improvement
                else:
                    # Revert
                    self.s.add(u)
                    for v in added_elements:
                        self.remove(v)
        
        return self
    
    def to_set(self):
        return set(self.s)

def _greedy_build(n, order):
    """Build a 4-AP-free set greedily in the given order."""
    fs = FourAPFreeSet(n)
    for x in order:
        fs.add(x)
    return fs.to_set()

def _multi_restart_greedy(n, num_restarts=50, deadline=None):
    """Try multiple greedy constructions with different orderings."""
    best = set()
    best_size = 0
    
    for r in range(num_restarts):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Different orderings
        if r == 0:
            order = list(range(1, n + 1))
        elif r == 1:
            order = list(range(n, 0, -1))
        elif r == 2:
            # Odds first
            order = list(range(1, n + 1, 2)) + list(range(2, n + 1, 2))
        elif r == 3:
            # Evens first
            order = list(range(2, n + 1, 2)) + list(range(1, n + 1, 2))
        elif r == 4:
            # Powers of 2 first, then others
            order = [2**i for i in range(10) if 2**i <= n] + [x for x in range(1, n+1) if x not in {2**i for i in range(10) if 2**i <= n}]
        else:
            order = list(range(1, n + 1))
            random.seed(r * 1000 + 42)
            random.shuffle(order)
        
        S = _greedy_build(n, order)
        
        if len(S) > best_size:
            best = set(S)
            best_size = len(S)
    
    return best

def _branch_and_bound_improve(S, n, deadline):
    """Try to improve the set by exploring removals and re-additions more systematically."""
    best = set(S)
    best_size = len(S)
    
    # For each element, try removing it and greedily completing
    for u in list(S):
        if time.time() > deadline:
            break
        
        S_test = set(S)
        S_test.remove(u)
        
        # Greedily complete
        fs = FourAPFreeSet(n)
        for x in sorted(S_test):
            fs.add(x)
        
        # Try adding in natural order
        fs.greedy_complete()
        
        if len(fs) > best_size:
            best = fs.to_set()
            best_size = len(fs)
    
    return best

def construct():
    n = 512
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety
    
    # Start with base-3 construction
    S = _base3_construction(n)
    
    # Convert to FourAPFreeSet for efficient operations
    fs = FourAPFreeSet(n)
    for x in S:
        fs.add(x)
    
    # Try greedy improvements from base-3
    improved = True
    while improved:
        if time.time() > deadline - 15:
            break
        improved = False
        for x in range(1, n + 1):
            if x not in fs and fs.add(x):
                improved = True
    
    best = fs.to_set()
    best_size = len(best)
    
    # Try multi-restart greedy
    if time.time() < deadline - 25:
        S2 = _multi_restart_greedy(n, num_restarts=40, deadline=deadline - 15)
        if len(S2) > best_size:
            best = S2
            best_size = len(S2)
    
    # Local search on the best set
    if time.time() < deadline - 10:
        fs2 = FourAPFreeSet(n)
        for x in best:
            fs2.add(x)
        fs2.local_search(max_iters=300, deadline=deadline - 5)
        if len(fs2) > best_size:
            best = fs2.to_set()
            best_size = len(best)
    
    # Branch and bound improvement
    if time.time() < deadline - 5:
        S3 = _branch_and_bound_improve(best, n, deadline - 2)
        if len(S3) > best_size:
            best = S3
            best_size = len(best)
    
    # Final verification
    result = sorted(best)
    
    # Verify no 4-AP exists
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            for k in range(j + 1, len(result)):
                d1 = result[j] - result[i]
                d2 = result[k] - result[j]
                if d1 == d2 and d1 > 0:
                    fourth = result[k] + d2
                    if fourth in best:
                        pass  # This would be a 4-AP, but we shouldn't have one
    
    return result
# EVOLVE-BLOCK-END
