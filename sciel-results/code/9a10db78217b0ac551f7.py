import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

N = 20
DEADLINE_FRACTION = 0.90

def _count_common_neighbors(masks, u, v, w):
    """Count common neighbors of u, v, w."""
    return (masks[u] & masks[v] & masks[w]).bit_count()

def _is_k33_free_masks(masks):
    """Check if graph with bitmask adjacency is K3,3-free."""
    for a in range(N):
        ma = masks[a]
        for b in range(a + 1, N):
            common_ab = ma & masks[b]
            if common_ab.bit_count() < 3:
                continue
            for c in range(b + 1, N):
                if (common_ab & masks[c]).bit_count() >= 3:
                    return False
    return True

def _edge_count_masks(masks):
    return sum(m.bit_count() for m in masks) // 2

def _can_add_edge_fast(masks, u, v):
    """
    Check if adding edge (u, v) keeps the graph K3,3-free.
    Only checks triples involving u or v.
    """
    # Temporarily add
    masks[u] |= (1 << v)
    masks[v] |= (1 << u)
    
    valid = True
    
    # Check triples (u, a, b)
    for a in range(N):
        if a == u: continue
        mau = masks[u] & masks[a]
        if mau.bit_count() < 2: # Need at least 2 common neighbors with u,a to potentially reach 3 with b? No.
            # If common(u,a) < 2, then common(u,a,b) <= common(u,a) < 3.
            # Wait, if common(u,a) is 2, and b is one of them, common is 2.
            # If common(u,a) is 1, common is 1.
            # So if common(u,a) < 2, no triple (u,a,b) can have 3 common neighbors?
            # Yes, because common(u,a,b) is subset of common(u,a).
            continue
        for b in range(a + 1, N):
            if b == u or b == v: continue # b cannot be u or v in the triple? 
            # The triple is vertices {u, a, b}. They must be distinct.
            # If b == v, then triple is {u, a, v}.
            # We check if |N(u) \cap N(a) \cap N(v)| >= 3.
            # Note: u is in N(v) and v is in N(u).
            # So u and v are common neighbors of themselves? No.
            # N(u) does not contain u.
            # So u is not in N(u) \cap N(a) \cap N(v).
            # v is in N(u). Is v in N(a)? Is v in N(v)? No.
            # So v is not a common neighbor of u, a, v.
            # So we just check the count.
            
            # Optimization: if b > a, we cover all pairs.
            # We need to check all a, b distinct from u.
            # The triple is {u, a, b}.
            # Common neighbors are in N(u) \cap N(a) \cap N(b).
            
            if (mau & masks[b]).bit_count() >= 3:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Revert
        masks[u] &= ~(1 << v)
        masks[v] &= ~(1 << u)
        return False
        
    # Check triples (v, a, b)
    for a in range(N):
        if a == v: continue
        mav = masks[v] & masks[a]
        if mav.bit_count() < 2:
            continue
        for b in range(a + 1, N):
            if b == v or b == u: continue
            if (mav & masks[b]).bit_count() >= 3:
                valid = False
                break
        if not valid:
            break
            
    # Revert
    masks[u] &= ~(1 << v)
    masks[v] &= ~(1 << u)
    
    return valid

def _repair_masks(masks):
    """Remove edges until graph is K3,3-free."""
    while not _is_k33_free_masks(masks):
        # Find a violation
        found = False
        for a in range(N):
            ma = masks[a]
            for b in range(a + 1, N):
                common_ab = ma & masks[b]
                if common_ab.bit_count() < 3:
                    continue
                for c in range(b + 1, N):
                    common_abc = common_ab & masks[c]
                    if common_abc.bit_count() >= 3:
                        # Violation found: {a,b,c} have >=3 common neighbors
                        # Pick a random common neighbor x and remove edge (a,x)
                        # This breaks the K3,3 for this triple
                        cn_list = [i for i in range(N) if common_abc & (1 << i)]
                        x = random.choice(cn_list)
                        # Remove edge (a, x)
                        masks[a] &= ~(1 << x)
                        masks[x] &= ~(1 << a)
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
    return masks

def _greedy_add_masks(masks):
    """Add all possible edges greedily."""
    improved = True
    while improved:
        improved = False
        for u in range(N):
            for v in range(u + 1, N):
                if not (masks[u] & (1 << v)):
                    if _can_add_edge_fast(masks, u, v):
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
                        improved = True
    return masks

def _directed_swap_search(masks, time_limit):
    """
    Try to add a non-edge by removing one existing edge.
    Returns new masks and count if improvement found, else None.
    """
    start_time = time.time()
    best_masks = masks[:]
    best_count = _edge_count_masks(best_masks)
    
    # Get lists of edges and non-edges
    edges = []
    non_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if masks[u] & (1 << v):
                edges.append((u, v))
            else:
                non_edges.append((u, v))
                
    # Shuffle to avoid bias
    random.shuffle(non_edges)
    random.shuffle(edges)
    
    for u_add, v_add in non_edges:
        if time.time() - start_time > time_limit:
            break
            
        # Check if we can add (u_add, v_add) directly
        if _can_add_edge_fast(masks, u_add, v_add):
            masks[u_add] |= (1 << v_add)
            masks[v_add] |= (1 << u_add)
            # Greedy fill after addition
            masks = _greedy_add_masks(masks)
            count = _edge_count_masks(masks)
            if count > best_count:
                best_count = count
                best_masks = masks[:]
            return best_masks, best_count
            
        # If not, try removing one edge to make it possible
        # We only need to consider edges that might be involved in the conflict.
        # But for simplicity, try all edges.
        # Optimization: If adding (u_add, v_add) creates violations, 
        # the violations must involve u_add or v_add.
        # So we only need to remove edges incident to u_add or v_add?
        # Not necessarily. Removing an edge (x,y) far away might not help.
        # But removing an edge (u_add, w) or (v_add, w) reduces common neighbors.
        # Let's try edges incident to u_add or v_add first.
        
        candidate_edges = []
        for u, v in edges:
            if u == u_add or v == u_add or u == v_add or v == v_add:
                candidate_edges.append((u, v))
                
        # If no incident edges, try all? No, unlikely to help.
        # Actually, removing an edge (x,y) where x,y are not u_add,v_add 
        # doesn't change N(u_add) or N(v_add). 
        # It changes N(x) and N(y).
        # The conflict is about common neighbors of triples involving u_add or v_add.
        # The common neighbors are in N(u_add) \cap N(a) \cap N(b).
        # Removing (x,y) only affects this if x or y is u_add, v_add, a, or b.
        # Since a,b are arbitrary, it's complex.
        # However, removing an edge incident to u_add or v_add is the most direct way 
        # to reduce the degree and thus the common neighbor counts.
        
        random.shuffle(candidate_edges)
        
        for u_rm, v_rm in candidate_edges:
            if time.time() - start_time > time_limit:
                break
                
            # Temporarily remove (u_rm, v_rm)
            masks[u_rm] &= ~(1 << v_rm)
            masks[v_rm] &= ~(1 << u_rm)
            
            # Check if (u_add, v_add) can be added now
            if _can_add_edge_fast(masks, u_add, v_add):
                # Add (u_add, v_add)
                masks[u_add] |= (1 << v_add)
                masks[v_add] |= (1 << u_add)
                
                # Greedy fill
                masks = _greedy_add_masks(masks)
                count = _edge_count_masks(masks)
                
                if count > best_count:
                    best_count = count
                    best_masks = masks[:]
                    return best_masks, best_count
                elif count == best_count:
                    # Accept to escape local optimum?
                    # We want to maximize, so we prefer improvement.
                    # But if we get stuck, we might need to move sideways.
                    # For now, let's require strict improvement or random acceptance.
                    if random.random() < 0.1:
                        best_count = count
                        best_masks = masks[:]
                        return best_masks, best_count
            else:
                # Revert removal
                masks[u_rm] |= (1 << v_rm)
                masks[v_rm] |= (1 << u_rm)
                
    return best_masks, best_count

def construct():
    """Construct a K3,3-free graph on 20 vertices maximizing edges."""
    start_time = time.time()
    deadline = start_time + 150 * DEADLINE_FRACTION
    
    best_masks = None
    best_count = 0
    
    # Initial random graph
    masks = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if random.random() < 0.5:
                masks[i] |= (1 << j)
                masks[j] |= (1 << i)
                
    masks = _repair_masks(masks)
    masks = _greedy_add_masks(masks)
    best_count = _edge_count_masks(masks)
    best_masks = masks[:]
    
    # Local search loop
    stagnant_iterations = 0
    max_stagnant = 50
    
    while time.time() < deadline:
        # Try directed swap
        new_masks, new_count = _directed_swap_search(masks, time_limit=5)
        
        if new_count > best_count:
            best_count = new_count
            best_masks = new_masks[:]
            masks = new_masks[:]
            stagnant_iterations = 0
        elif new_count == best_count and new_masks != masks:
            # Sideways move
            if random.random() < 0.5:
                masks = new_masks[:]
                stagnant_iterations += 1
            else:
                stagnant_iterations += 1
        else:
            stagnant_iterations += 1
            
        # If stuck, perturb
        if stagnant_iterations > max_stagnant:
            # Remove random edges
            for _ in range(random.randint(3, 10)):
                edges = []
                for i in range(N):
                    for j in range(i + 1, N):
                        if masks[i] & (1 << j):
                            edges.append((i, j))
                if not edges:
                    break
                u, v = random.choice(edges)
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
                
            masks = _repair_masks(masks)
            masks = _greedy_add_masks(masks)
            count = _edge_count_masks(masks)
            if count > best_count:
                best_count = count
                best_masks = masks[:]
            stagnant_iterations = 0
            max_stagnant = random.randint(20, 100) # Annealing-like

    # Final greedy
    if best_masks is not None and time.time() < deadline - 5:
        best_masks = _greedy_add_masks(best_masks)
        best_count = _edge_count_masks(best_masks)
        
    if best_masks is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    edges = []
    for i in range(N):
        m = best_masks[i]
        for j in range(i + 1, N):
            if m & (1 << j):
                edges.append([i, j])
                
    return edges

# EVOLVE-BLOCK-END
