import time
import random

class _DPState:
    """Maintains cardinality-indexed subset-sum DP state incrementally."""
    def __init__(self, arr):
        self.arr = list(arr)
        self.n = len(arr)
        self.cur_set = set(arr)
        self._rebuild()
    
    def _rebuild(self):
        n = self.n
        self.dp = [0] * (n + 1)
        self.dp[0] = 1
        for x in self.arr:
            for k in range(n, 0, -1):
                self.dp[k] |= (self.dp[k - 1] << x)
        self.seen = 0
        for k in range(n + 1):
            self.seen |= self.dp[k]
    
    def is_valid(self):
        # Check no sum appears in two different cardinalities
        seen = 0
        for k in range(self.n + 1):
            if self.dp[k] & seen:
                return False
            seen |= self.dp[k]
        return True
    
    def try_add(self, candidate):
        """Try adding candidate. Returns True if valid, updates state if so."""
        if candidate in self.cur_set:
            return False
        new_n = self.n + 1
        if new_n > 60:
            return False
        # Compute new dp
        new_dp = [0] * (new_n + 1)
        new_dp[0] = 1
        for x in self.arr:
            for k in range(new_n, 0, -1):
                new_dp[k] |= (new_dp[k - 1] << x)
        for k in range(new_n, 0, -1):
            new_dp[k] |= (new_dp[k - 1] << candidate)
        # Check validity
        seen = 0
        for k in range(new_n + 1):
            if new_dp[k] & seen:
                return False
            seen |= new_dp[k]
        # Commit
        self.arr.append(candidate)
        self.cur_set.add(candidate)
        self.n = new_n
        self.dp = new_dp
        return True
    
    def get_candidates_by_risk(self, all_candidates):
        """Rank candidates by collision risk (lower is better)."""
        n = self.n
        # For each candidate, estimate how many new sums it creates that might collide
        scored = []
        for c in all_candidates:
            if c in self.cur_set:
                continue
            # Estimate: for each k, new sums = dp[k-1] << c
            # Collision risk = overlap with existing seen sums at different k
            risk = 0
            for k in range(1, n + 1):
                new_sums = self.dp[k - 1] << c
                # Check overlap with seen sums at other cardinalities
                # A sum s in new_sums at cardinality k collides if s is in seen
                # and s was achieved at a different cardinality
                # Simple heuristic: count bits in new_sums & seen
                risk += bin(new_sums & self.seen).count('1')
            scored.append((risk, c))
        scored.sort()
        return [c for _, c in scored]

def _is_valid(arr):
    """Check: distinct ints in 1..256, size 1..60, cardinality-separated subset sums."""
    n = len(arr)
    if n < 1 or n > 60:
        return False
    s = set()
    for v in arr:
        if not isinstance(v, int) or isinstance(v, bool):
            return False
        if v < 1 or v > 256:
            return False
        if v in s:
            return False
        s.add(v)
    state = _DPState(arr)
    return state.is_valid()

def _grow_greedy_risk(start, deadline, rng=None):
    """Greedy growth using risk-based candidate ordering."""
    if rng is None:
        rng = random.Random(42)
    state = _DPState(start)
    if not state.is_valid():
        return list(start)
    
    all_candidates = [c for c in range(1, 257) if c not in state.cur_set]
    
    while time.monotonic() < deadline:
        if not all_candidates:
            break
        # Get candidates sorted by risk
        ordered = state.get_candidates_by_risk(all_candidates)
        added = False
        for c in ordered:
            if c in state.cur_set:
                continue
            if state.try_add(c):
                all_candidates.remove(c)
                added = True
                break
        if not added:
            # Try random order as fallback
            shuffled = all_candidates[:]
            rng.shuffle(shuffled)
            for c in shuffled:
                if c in state.cur_set:
                    continue
                if state.try_add(c):
                    all_candidates.remove(c)
                    added = True
                    break
            if not added:
                break
    return state.arr

def _local_search(cur, deadline, rng):
    """Try to improve by swapping elements out for better ones."""
    best = list(cur)
    best_size = len(cur)
    
    while time.monotonic() < deadline - 3:
        improved = False
        cur_set = set(cur)
        for i in range(len(cur)):
            if time.monotonic() > deadline - 3:
                break
            for c in range(1, 257):
                if c in cur_set:
                    continue
                new_cur = cur[:i] + [c] + cur[i+1:]
                if _is_valid(new_cur):
                    grown = _grow_greedy_risk(new_cur, min(deadline - 3, time.monotonic() + 8), rng)
                    if len(grown) > best_size:
                        best = grown
                        best_size = len(grown)
                        cur = list(grown)
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    return best

def construct():
    start_time = time.monotonic()
    deadline = start_time + 85.0  # safety margin for 90s limit
    
    rng = random.Random(42)
    
    start_strategies = [
        [1],
        [1, 2],
        [1, 2, 4],
        [1, 2, 4, 8],
        [1, 2, 4, 8, 16],
        [1, 2, 4, 8, 16, 32],
        [1, 2, 4, 8, 16, 32, 64],
        [1, 2, 4, 8, 16, 32, 64, 128],
        [2, 4, 8, 16, 32, 64, 128],
        [1, 3, 7, 15, 31, 63, 127],
        [2, 6, 12, 24, 48, 96],
        [10, 20, 40, 80, 160],
        [1, 2, 5, 10, 20, 40, 80, 160],
        [50, 100, 150],
        [40, 80, 120, 160],
        [30, 60, 90, 120, 150, 180, 210, 240],
    ]
    
    best_result = None
    best_size = 0
    
    for start in start_strategies:
        if time.monotonic() > deadline - 5:
            break
        if not _is_valid(start):
            continue
        
        grown = _grow_greedy_risk(start, deadline, rng)
        if len(grown) > best_size:
            best_size = len(grown)
            best_result = grown
        
        if time.monotonic() < deadline - 10:
            improved = _local_search(grown, deadline, rng)
            if len(improved) > best_size:
                best_size = len(improved)
                best_result = improved
    
    # Final attempt: random starts
    for _ in range(5):
        if time.monotonic() > deadline - 3:
            break
        random_start = [rng.randint(1, 256) for _ in range(rng.randint(1, 5))]
        random_start = list(set(random_start))
        if not _is_valid(random_start):
            continue
        grown = _grow_greedy_risk(random_start, deadline, rng)
        if len(grown) > best_size:
            best_size = len(grown)
            best_result = grown
    
    if best_result is None:
        return list(range(1, 9))
    
    return list(best_result)
