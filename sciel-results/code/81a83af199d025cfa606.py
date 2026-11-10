"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time

# EVOLVE-BLOCK-START

def _calc_alpha(triples):
    """Compute independence number of the 3-uniform hypergraph on 16 points."""
    n = 16
    # Bitmask for each triple
    masks = []
    for t in triples:
        m = 0
        for v in t:
            m |= (1 << v)
        masks.append(m)
    
    # Find largest independent set via branch and bound
    best = [0]
    best_set = [0]
    
    def search(remaining_mask, current_mask, current_size):
        if current_size > best[0]:
            best[0] = current_size
            best_set[0] = current_mask
        
        # Upper bound: current_size + popcount(remaining_mask)
        if current_size + bin(remaining_mask).count('1') <= best[0]:
            return
        
        # Try to add vertices from remaining_mask
        while remaining_mask:
            # Pick the lowest set bit
            lb = remaining_mask & (-remaining_mask)
            v = lb.bit_length() - 1
            remaining_mask ^= lb
            
            # Check if adding v keeps independence
            new_mask = current_mask | lb
            independent = True
            for m in masks:
                if (new_mask & m) == m:
                    independent = False
                    break
            
            if independent:
                search(remaining_mask, new_mask, current_size + 1)
    
    search((1 << 16) - 1, 0, 0)
    return best[0]


def _is_linear(triples):
    """Check if any two triples share at most one point."""
    n = len(triples)
    for i in range(n):
        for j in range(i + 1, n):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True


def _build_maximal_linear():
    """Build a maximal linear 3-uniform hypergraph on 16 points."""
    n = 16
    all_points = list(range(n))
    triples = []
    used_pairs = set()
    
    # Generate all possible triples sorted by a heuristic
    candidates = []
    for a, b, c in itertools.combinations(range(n), 3):
        # Score: prefer triples that use "spread out" points
        # Use modular distance as a proxy for balance
        score = (b - a) + (c - b) + (a - c + n)  # not quite right, use different
        # Better: use sum of pairwise distances
        d1 = min(abs(a-b), n - abs(a-b))
        d2 = min(abs(b-c), n - abs(b-c))
        d3 = min(abs(a-c), n - abs(a-c))
        score = d1 + d2 + d3
        candidates.append((score, a, b, c))
    
    # Sort by score (higher = more balanced), then by tuple for determinism
    candidates.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    
    for score, a, b, c in candidates:
        # Check pair constraints
        pairs = [(a, b), (a, c), (b, c)]
        valid = True
        for p in pairs:
            p_sorted = tuple(sorted(p))
            if p_sorted in used_pairs:
                valid = False
                break
        
        if valid:
            triples.append([a, b, c])
            for p in pairs:
                used_pairs.add(tuple(sorted(p)))
    
    return triples


def _greedy_reduce_alpha(triples, time_limit):
    """Try to reduce independence number by removing edges that cover many uncovered vertices."""
    n = 16
    deadline = time.time() + time_limit
    
    current = list(triples)
    current_alpha = _calc_alpha(current)
    best = list(current)
    best_alpha = current_alpha
    
    # Precompute masks for current triples
    masks = []
    for t in current:
        m = 0
        for v in t:
            m |= (1 << v)
        masks.append(m)
    
    # Find independent set for current
    def find_independent_set(masks):
        """Find a maximum independent set (for greedy heuristic)."""
        best_size = 0
        best_set = 0
        remaining = (1 << n) - 1
        
        # Greedy: repeatedly add vertex that blocks fewest other vertices
        chosen = 0
        remaining = (1 << n) - 1
        
        while remaining:
            best_v = -1
            best_conflicts = -1
            for v in range(n):
                lb = 1 << v
                if not (remaining & lb):
                    continue
                # Count how many remaining vertices would be blocked
                conflicts = 0
                new_mask = chosen | lb
                for m in masks:
                    if (new_mask & m) == m:
                        # This triple would be fully covered, so any vertex in it is blocked
                        for u in range(n):
                            if (m >> u) & 1 and (remaining >> u) & 1:
                                conflicts += 1
                        break
                if best_conflicts < 0 or conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_v = v
            
            if best_v < 0:
                break
            
            chosen |= (1 << best_v)
            remaining &= ~(1 << best_v)
        
        return chosen
    
    # Try removing edges one at a time
    while time.time() < deadline:
        improved = False
        # Try each edge removal
        for i in range(len(current)):
            if time.time() >= deadline:
                break
            
            # Remove edge i
            trial = current[:i] + current[i+1:]
            
            # Quick check: does removing this edge reduce alpha?
            # Compute alpha of trial
            trial_alpha = _calc_alpha(trial)
            
            if trial_alpha < current_alpha:
                current = trial
                current_alpha = trial_alpha
                improved = True
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best = list(current)
                break
        
        if not improved:
            break
    
    return best, best_alpha


def construct():
    """Construct a linear triple system on 16 points with small independence number."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Build initial maximal linear hypergraph
    triples = _build_maximal_linear()
    
    if not _is_linear(triples):
        # Fallback to trivial
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    # Get initial alpha
    initial_alpha = _calc_alpha(triples)
    
    # Try to reduce alpha with time budget
    time_remaining = deadline - time.time()
    if time_remaining > 5:
        triples, alpha = _greedy_reduce_alpha(triples, time_remaining - 5)
    else:
        alpha = initial_alpha
    
    # Final validity check
    if not _is_linear(triples):
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return triples

# EVOLVE-BLOCK-END
