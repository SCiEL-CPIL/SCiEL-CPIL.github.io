"""New program: Hypergraph SA with entropy penalty for diverse valid colorings."""
import random
import math
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
NUM_COLORS = 120

def _build_edge_index():
    """Map (i,j) with i<j to a linear index 0..119."""
    edge_to_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = idx
            idx += 1
    return edge_to_idx

def _build_cycles():
    """
    For each 4-vertex subset, compute the 3 cycles.
    Each cycle is a list of 4 edge indices.
    Return list of cycles (each is a tuple of 4 edge indices).
    """
    edge_to_idx = _build_edge_index()
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = edge_to_idx[(min(a,b), max(a,b))]
        e2 = edge_to_idx[(min(b,c), max(b,c))]
        e3 = edge_to_idx[(min(c,d), max(c,d))]
        e4 = edge_to_idx[(min(d,a), max(d,a))]
        cycles.append((e1, e2, e3, e4))
        
        # Cycle 2: a-b-d-c-a
        e1 = edge_to_idx[(min(a,b), max(a,b))]
        e2 = edge_to_idx[(min(b,d), max(b,d))]
        e3 = edge_to_idx[(min(d,c), max(d,c))]
        e4 = edge_to_idx[(min(c,a), max(c,a))]
        cycles.append((e1, e2, e3, e4))
        
        # Cycle 3: a-c-b-d-a
        e1 = edge_to_idx[(min(a,c), max(a,c))]
        e2 = edge_to_idx[(min(c,b), max(c,b))]
        e3 = edge_to_idx[(min(b,d), max(b,d))]
        e4 = edge_to_idx[(min(d,a), max(d,a))]
        cycles.append((e1, e2, e3, e4))
    return cycles

def _build_edge_to_cycles():
    """Map each edge index to list of cycle indices it participates in."""
    cycles = _build_cycles()
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for ei in cycle:
            edge_to_cycles[ei].append(ci)
    return edge_to_cycles

# Precompute
_EDGE_TO_IDX = _build_edge_index()
_CYCLES = _build_cycles()
_EDGE_TO_CYCLES = _build_edge_to_cycles()
NUM_CYCLES = len(_CYCLES)

def _compute_entropy(color_counts):
    """Compute Shannon entropy of color distribution. Higher = more uniform/diverse."""
    total = sum(color_counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in color_counts:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy

def construct():
    random.seed(42)
    
    # Initialize with all same color (valid, low diversity)
    colors = [0] * NUM_EDGES
    color_counts = [0] * NUM_COLORS
    color_counts[0] = NUM_EDGES
    
    # Track state
    violations = 0  # count of bad cycles (4 distinct colors)
    
    # Helper: check if a cycle is bad
    def is_bad_cycle(ci):
        c = _CYCLES[ci]
        c0, c1, c2, c3 = colors[c[0]], colors[c[1]], colors[c[2]], colors[c[3]]
        return len({c0, c1, c2, c3}) == 4
    
    # Initial violations (should be 0)
    for ci in range(NUM_CYCLES):
        if is_bad_cycle(ci):
            violations += 1
    
    best_colors = colors[:]
    best_distinct = sum(1 for c in color_counts if c > 0)
    
    # Simulated annealing parameters
    T = 50.0
    T_min = 0.05
    cooling = 0.995
    max_iter = 80000
    
    # Cost function: violations * W_V - entropy * W_E
    # We want to minimize cost. Entropy is maximized when colors are spread out.
    W_V = 1000  # Strong penalty for violations
    W_E = 1.0   # Weight for entropy term
    
    def get_cost():
        ent = _compute_entropy(color_counts)
        return violations * W_V - ent * W_E
    
    current_cost = get_cost()
    
    for iteration in range(max_iter):
        if T < T_min:
            break
            
        # Pick a random edge
        ei = random.randint(0, NUM_EDGES - 1)
        old_color = colors[ei]
        
        # Choose new color with bias towards less-used colors to encourage diversity
        if random.random() < 0.6:
            # Pick a rarely used or unused color
            candidate_colors = [c for c in range(NUM_COLORS) if color_counts[c] < 5]
            if not candidate_colors:
                candidate_colors = list(range(NUM_COLORS))
            new_color = random.choice(candidate_colors)
        else:
            # Pick random color
            new_color = random.randint(0, NUM_COLORS - 1)
            
        if new_color == old_color:
            continue
            
        # Calculate delta violations
        delta_violations = 0
        for ci in _EDGE_TO_CYCLES[ei]:
            c = _CYCLES[ci]
            # Old state
            old_set = set()
            for e in c:
                if e == ei:
                    old_set.add(old_color)
                else:
                    old_set.add(colors[e])
            old_bad = (len(old_set) == 4)
            
            # New state
            new_set = set()
            for e in c:
                if e == ei:
                    new_set.add(new_color)
                else:
                    new_set.add(colors[e])
            new_bad = (len(new_set) == 4)
            
            if old_bad and not new_bad:
                delta_violations -= 1
            elif not old_bad and new_bad:
                delta_violations += 1
        
        # Calculate delta entropy
        old_entropy = _compute_entropy(color_counts)
        
        # Temporarily update counts
        color_counts[old_color] -= 1
        color_counts[new_color] += 1
        new_entropy = _compute_entropy(color_counts)
        
        # Revert for now (will apply if accepted)
        color_counts[old_color] += 1
        color_counts[new_color] -= 1
        
        delta_entropy = new_entropy - old_entropy
        
        # Delta cost
        delta_cost = delta_violations * W_V - delta_entropy * W_E
        
        # Acceptance
        accept = False
        if delta_cost < 0:
            accept = True
        else:
            # Avoid overflow
            if delta_cost / T > 500:
                prob = 0.0
            else:
                prob = math.exp(-delta_cost / T)
            if random.random() < prob:
                accept = True
                
        if accept:
            # Apply change
            colors[ei] = new_color
            color_counts[old_color] -= 1
            color_counts[new_color] += 1
            violations += delta_violations
            current_cost += delta_cost
            
            # Track best valid solution
            if violations == 0:
                distinct = sum(1 for c in color_counts if c > 0)
                if distinct > best_distinct:
                    best_distinct = distinct
                    best_colors = colors[:]
        
        T *= cooling
    
    # Final: ensure we return a valid solution
    return best_colors
