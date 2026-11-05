import random
import time

N = 16
NUM_EDGES = N * (N - 1) // 2
NUM_COLORS = 120

def _get_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def _get_edges():
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

def _get_c4s():
    """Get all 4-cycles as tuples of 4 edge indices."""
    c4s = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Three 4-cycles on {a,b,c,d}
                    # Cycle 1: a-b-c-d-a: edges (a,b),(b,c),(c,d),(d,a)
                    # Cycle 2: a-b-d-c-a: edges (a,b),(b,d),(d,c),(c,a)
                    # Cycle 3: a-c-b-d-a: edges (a,c),(c,b),(b,d),(d,a)
                    c4s.append((_get_idx(a,b), _get_idx(b,c), _get_idx(c,d), _get_idx(d,a)))
                    c4s.append((_get_idx(a,b), _get_idx(b,d), _get_idx(d,c), _get_idx(c,a)))
                    c4s.append((_get_idx(a,c), _get_idx(c,b), _get_idx(b,d), _get_idx(d,a)))
    return c4s

def _count_colors(colors):
    return len(set(colors))

def _is_valid(colors):
    """Check if no 4-cycle has 4 distinct colors."""
    for e1, e2, e3, e4 in _C4S:
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
            return False
    return True

def _check_c4_valid(colors, e1, e2, e3, e4):
    """Check if a specific 4-cycle is valid."""
    c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
    if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
        return False
    return True

# Precompute all 4-cycles
_C4S = _get_c4s()
_EDGES = _get_edges()

# For each edge, precompute which 4-cycles it belongs to
_EDGE_TO_C4S = [[] for _ in range(NUM_EDGES)]
for idx, (e1, e2, e3, e4) in enumerate(_C4S):
    for e in (e1, e2, e3, e4):
        _EDGE_TO_C4S[e].append(idx)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_score = 0
    best_colors = [0] * NUM_EDGES
    
    random.seed(42)
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Try different random orderings
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        colors = [0] * NUM_EDGES
        used_colors = set([0])
        # Start with all 0, then try to assign new colors greedily
        
        # Reset: start fresh
        colors = [0] * NUM_EDGES
        used_colors = set()
        
        # First, assign color 0 to all
        # Then greedily try to change edges to new colors
        
        # Better: start with all 0, then for each edge in random order,
        # try to assign a new color if valid, else keep/try existing
        
        # Actually, let's do: assign all 0 first, then try to upgrade edges
        # to new colors while maintaining validity
        
        # Start: all edges color 0
        colors = [0] * NUM_EDGES
        used_colors = {0}
        
        # For each edge in random order, try to assign a new color
        for edge_idx in edge_order:
            # Try to find a new color that keeps validity
            # Check all 4-cycles containing this edge
            relevant_c4s = _EDGE_TO_C4S[edge_idx]
            
            # Try new colors first (highest unused), then existing
            # To maximize diversity, try unused colors first
            # But we need to check validity
            
            # Collect colors that are forbidden
            # A color c is forbidden if assigning it to edge_idx would create
            # a rainbow 4-cycle in any relevant c4
            
            # For efficiency, just try a few candidates
            # Try: a new color (not in used_colors), or an existing color
            
            # Try new colors: pick from unused
            found_new = False
            # Try a random unused color
            unused = [c for c in range(NUM_COLORS) if c not in used_colors]
            random.shuffle(unused)
            
            # Limit checks for speed
            max_tries = min(len(unused), 20)
            for trial in range(max_tries):
                new_color = unused[trial]
                colors[edge_idx] = new_color
                
                # Check all relevant 4-cycles
                valid = True
                for c4_idx in relevant_c4s:
                    e1, e2, e3, e4 = _C4S[c4_idx]
                    c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
                    if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
                        valid = False
                        break
                
                if valid:
                    used_colors.add(new_color)
                    found_new = True
                    break
            
            if not found_new:
                # Revert to 0 or try existing colors
                # Try existing colors
                existing = list(used_colors)
                random.shuffle(existing)
                max_tries2 = min(len(existing), 10)
                found_existing = False
                for trial in range(max_tries2):
                    old_color = existing[trial]
                    colors[edge_idx] = old_color
                    
                    valid = True
                    for c4_idx in relevant_c4s:
                        e1, e2, e3, e4 = _C4S[c4_idx]
                        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
                        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
                            valid = False
                            break
                    
                    if valid:
                        found_existing = True
                        break
                
                if not found_existing:
                    colors[edge_idx] = 0  # fallback
        
        # Count colors used
        score = len(used_colors)
        if score > best_score:
            best_score = score
            best_colors = colors[:]
        
        # If we used all 120 colors, we're done
        if best_score == NUM_COLORS:
            break
    
    return best_colors
