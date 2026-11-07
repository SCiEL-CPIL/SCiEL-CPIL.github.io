import time
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    n = 12
    period = 27720  # lcm(1..12)
    max_L = 144
    
    # Precompute: valid_offsets[i][rem] = list of d in 1..max_L where d % i == rem
    valid_offsets = {}
    for i in range(1, n + 1):
        valid_offsets[i] = [[] for _ in range(i)]
        for d in range(1, max_L + 1):
            rem = d % i
            valid_offsets[i][rem].append(d)
    
    def count_options(m, L):
        """Count valid d's for each i given m and L."""
        counts = []
        for i in range(1, n + 1):
            rem = (-m) % i
            cnt = 0
            for d in valid_offsets[i][rem]:
                if d > L:
                    break
                cnt += 1
            counts.append(cnt)
        return counts
    
    def has_perfect_matching(m, L):
        """Check if (m, m+L] has a perfect matching for distinct multiples."""
        if L < n:
            return False
        
        # For each i, compute the valid d's up to L and count them
        # Order left nodes by number of options (most constrained first)
        adj = []  # adj[i-1] = list of valid d's for left node i
        counts = []
        for i in range(1, n + 1):
            rem = (-m) % i
            opts = []
            for d in valid_offsets[i][rem]:
                if d > L:
                    break
                opts.append(d)
            adj.append(opts)
            counts.append(len(opts))
        
        # Check if any node has 0 options
        for c in counts:
            if c == 0:
                return False
        
        # Hall's condition quick check: for any subset S of left nodes,
        # |N(S)| >= |S|. We check small subsets as a heuristic.
        # Actually, let's just do the full matching.
        
        # Order left nodes by count (ascending) - most constrained first
        order = sorted(range(n), key=lambda idx: counts[idx])
        
        # matchR[d] = index in original left set (0-based) that d is matched to
        matchR = [-1] * (L + 1)
        
        def dfs(idx, seen):
            # idx is the original left node index (0-based)
            for d in adj[idx]:
                if seen[d]:
                    continue
                seen[d] = True
                if matchR[d] == -1 or dfs(matchR[d], seen):
                    matchR[d] = idx
                    return True
            return False
        
        for orig_idx in order:
            seen = [False] * (L + 1)
            if not dfs(orig_idx, seen):
                return False
        return True
    
    def find_min_L(m):
        """Binary search for minimum L for a given m."""
        lo, hi = 1, max_L
        min_L_for_m = max_L
        
        while lo <= hi:
            mid = (lo + hi) // 2
            if has_perfect_matching(m, mid):
                min_L_for_m = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return min_L_for_m
    
    best_m = 0
    best_L = 0
    
    # Phase 1: Smart candidate selection
    # Score m by how "bad" its residue coverage is at small L values.
    # An m is good if for some small L, at least one i has very few options.
    # We want to find m that forces large L.
    
    # Precompute a heuristic score for each m in the period
    # For each m, compute the minimum count over all i at L=12, L=15, etc.
    # Lower minimum count at small L suggests larger f(12,m).
    
    # Actually, let's just scan all m but be smart about ordering.
    # Start with m values that are likely to have large f.
    
    # Heuristic: m is "bad" (good for us) if for many small L,
    # some i has 0 or 1 valid d in (m, m+L].
    
    # Let's compute a score for each m
    scores = {}
    for m in range(period):
        score = 0
        for L_check in [12, 14, 16, 18, 20, 24, 30, 40, 50, 60, 80, 100, 120, 144]:
            min_cnt = min(count_options(m, L_check))
            # Lower min_cnt is better for us (forces larger L)
            score += (15 - min_cnt) * 10
        scores[m] = score
    
    # Sort m values by score descending (most promising first)
    sorted_ms = sorted(range(period), key=lambda m: scores[m], reverse=True)
    
    # Phase 2: Evaluate candidates in order of promise
    evaluated = 0
    for m in sorted_ms:
        if time.time() - start_time > deadline:
            break
        
        min_L_for_m = find_min_L(m)
        evaluated += 1
        
        if min_L_for_m > best_L:
            best_L = min_L_for_m
            best_m = m
        
        # Early stopping: if we've found L=144, that's the max possible
        if best_L >= max_L:
            break
    
    # Phase 3: If time remains, do a full scan to make sure we didn't miss anything
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 5.0 and best_L < max_L:
        # Check m values we haven't evaluated yet
        evaluated_set = set(sorted_ms[:evaluated])
        for m in range(period):
            if time.time() - start_time > deadline:
                break
            if m in evaluated_set:
                continue
            min_L_for_m = find_min_L(m)
            if min_L_for_m > best_L:
                best_L = min_L_for_m
                best_m = m
            if best_L >= max_L:
                break
    
    if best_m < 0 or best_m > 1000000:
        best_m = 0
    
    return int(best_m)
