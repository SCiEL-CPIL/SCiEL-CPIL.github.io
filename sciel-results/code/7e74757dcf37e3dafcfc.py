"""Dense odd-order graphs avoiding equal-degree path endpoints."""
import time
import random

# EVOLVE-BLOCK-START
N = 15
DEADLINE = 85.0  # safety margin before 90s kill
TOTAL_EDGES = N * (N - 1) // 2

# Precompute edge index mapping
EDGE_INDEX = {}
_idx = 0
for _u in range(N):
    for _v in range(_u + 1, N):
        EDGE_INDEX[(_u, _v)] = _idx
        _idx += 1

def _bitmask_to_edges(mask):
    """Convert bitmask to edge list."""
    edges = []
    for (u, v), idx in EDGE_INDEX.items():
        if mask & (1 << idx):
            edges.append([u, v])
    return edges

def _get_degrees_from_mask(mask):
    """Compute degree of each vertex from bitmask."""
    deg = [0] * N
    for (u, v), idx in EDGE_INDEX.items():
        if mask & (1 << idx):
            deg[u] += 1
            deg[v] += 1
    return deg

def _is_valid_mask(mask):
    """Check if no 3-edge path s-u-v-t has deg(s)==deg(t)."""
    deg = _get_degrees_from_mask(mask)
    
    # Build adjacency list
    adj = [[] for _ in range(N)]
    for (u, v), idx in EDGE_INDEX.items():
        if mask & (1 << idx):
            adj[u].append(v)
            adj[v].append(u)
    
    # For each edge (u,v), check all s in N(u)\{v} and t in N(v)\{u}, s!=t
    for (u, v), idx in EDGE_INDEX.items():
        if not (mask & (1 << idx)):
            continue
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if deg[s] == deg[t]:
                    return False
    return True

def _edge_count(mask):
    return bin(mask).count('1')

def _try_add_edge(mask, u, v):
    """Try adding edge (u,v); return new mask if valid, else None."""
    if u == v:
        return None
    if u > v:
        u, v = v, u
    idx = EDGE_INDEX.get((u, v))
    if idx is None:
        return None
    if mask & (1 << idx):
        return mask
    
    new_mask = mask | (1 << idx)
    if _is_valid_mask(new_mask):
        return new_mask
    return None

def construct():
    start_time = time.time()
    random.seed(42)
    
    best_mask = 0
    best_count = 0
    
    # Greedy construction
    all_edges = [(u, v) for u in range(N) for v in range(u+1, N)]
    random.shuffle(all_edges)
    
    current_mask = 0
    for u, v in all_edges:
        if time.time() - start_time > DEADLINE:
            break
        new_mask = _try_add_edge(current_mask, u, v)
        if new_mask is not None:
            current_mask = new_mask
    
    best_mask = current_mask
    best_count = _edge_count(current_mask)
    
    # Local search
    for iteration in range(500):
        if time.time() - start_time > DEADLINE:
            break
        
        improved = False
        
        # Try adding each missing edge
        for (u, v), idx in EDGE_INDEX.items():
            if current_mask & (1 << idx):
                continue
            new_mask = current_mask | (1 << idx)
            if _is_valid_mask(new_mask):
                current_mask = new_mask
                best_mask = current_mask
                best_count = _edge_count(current_mask)
                improved = True
                break
        
        if improved:
            continue
        
        # Try 2-flips
        edges_present = [(u, v) for (u, v), idx in EDGE_INDEX.items() if current_mask & (1 << idx)]
        edges_missing = [(u, v) for (u, v), idx in EDGE_INDEX.items() if not (current_mask & (1 << idx))]
        
        for _ in range(100):
            if time.time() - start_time > DEADLINE:
                break
            if not edges_present or not edges_missing:
                break
            
            e1 = random.choice(edges_present)
            e2 = random.choice(edges_missing)
            
            idx1 = EDGE_INDEX[e1]
            idx2 = EDGE_INDEX[e2]
            
            mask_no_e1 = current_mask & ~(1 << idx1)
            new_mask = mask_no_e1 | (1 << idx2)
            
            if new_mask != current_mask and _is_valid_mask(new_mask):
                current_mask = new_mask
                best_mask = current_mask
                best_count = _edge_count(current_mask)
                improved = True
                break
        
        if not improved and edges_present:
            # Perturbation
            to_remove = random.sample(edges_present, min(2, len(edges_present)))
            pert_mask = current_mask
            for e in to_remove:
                pert_mask &= ~(1 << EDGE_INDEX[e])
            
            # Greedily re-add
            re_add_edges = [(u, v) for (u, v), idx in EDGE_INDEX.items() if not (pert_mask & (1 << idx))]
            random.shuffle(re_add_edges)
            for u, v in re_add_edges:
                if time.time() - start_time > DEADLINE:
                    break
                nm = _try_add_edge(pert_mask, u, v)
                if nm is not None:
                    pert_mask = nm
            
            if _edge_count(pert_mask) >= best_count:
                current_mask = pert_mask
                best_mask = pert_mask
                best_count = _edge_count(current_mask)
    
    result = _bitmask_to_edges(best_mask)
    result = [[int(u), int(v)] for u, v in result]
    return result
# EVOLVE-BLOCK-END
