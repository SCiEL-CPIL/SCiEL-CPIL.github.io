"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def _calc_alpha_brute(triples, n=16):
    """Calculate the independence number using brute force bitmasks."""
    if not triples:
        return n
    
    # Precompute triple masks
    triple_masks = []
    for a, b, c in triples:
        triple_masks.append((1 << a) | (1 << b) | (1 << c))
    
    # Try all subsets of size k from largest to smallest
    for k in range(n, 0, -1):
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            # Check if any triple is fully contained
            contained = False
            for tm in triple_masks:
                if (tm & mask) == tm:
                    contained = True
                    break
            if not contained:
                return k
    return 0

def _calc_alpha_ilp(triples, n=16):
    """Calculate the independence number using Integer Linear Programming."""
    if not triples:
        return n
        
    # Variables x_0 ... x_15, x_i in {0, 1}
    # Maximize sum(x_i)
    # Subject to: x_a + x_b + x_c <= 2 for each triple (a, b, c)
    
    c = -np.ones(n) # Minimize -sum(x)
    
    A_ub_list = []
    b_ub_list = []
    
    for a, b, c_ in triples:
        row = np.zeros(n)
        row[a] = 1
        row[b] = 1
        row[c_] = 1
        A_ub_list.append(row)
        b_ub_list.append(2)
        
    if not A_ub_list:
        return n
        
    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)
    
    constraints = [LinearConstraint(A_ub, -np.inf, 2)]
    bounds = Bounds(0, 1)
    integrality = np.ones(n)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.status == 0:
            # res.x is the solution
            alpha_val = int(round(sum(res.x)))
            # Sanity check: alpha cannot be larger than n
            if alpha_val > n:
                return _calc_alpha_brute(triples, n)
            return alpha_val
        else:
            # Fallback if ILP fails
            return _calc_alpha_brute(triples, n)
    except Exception:
        return _calc_alpha_brute(triples, n)

def _generate_random_linear_graph(n=16):
    """Generate a random maximal linear 3-graph."""
    all_triples = list(itertools.combinations(range(n), 3))
    random.shuffle(all_triples)
    
    triples = []
    used_pairs = set()
    
    for t in all_triples:
        a, b, c = t
        pairs = [(a, b), (a, c), (b, c)]
        # Check if any pair is already used
        if any(p in used_pairs for p in pairs):
            continue
        
        triples.append(t)
        for p in pairs:
            used_pairs.add(p)
            
    return triples

def _check_linearity(triples):
    """Verify if the list of triples is linear."""
    pairs_seen = set()
    for t in triples:
        a, b, c = t
        pairs = [(a, b), (a, c), (b, c)]
        for p in pairs:
            if p in pairs_seen:
                return False
            pairs_seen.add(p)
    return True

def construct():
    """Construct a linear 3-graph on 16 points with small independence number."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 16
    best_triples = []
    best_alpha = n + 1
    
    # Ensure we have a valid fallback
    fallback_triples = _generate_random_linear_graph(n)
    if fallback_triples:
        best_triples = fallback_triples
        best_alpha = _calc_alpha_ilp(best_triples, n)
    else:
        # Absolute fallback: empty graph (alpha=16)
        best_triples = []
        best_alpha = 16

    # Local Search
    while time.time() < deadline:
        # Restart with a new random graph if we've been stuck or just starting a new phase
        # We'll just generate a new random graph every few seconds or if stuck
        # For now, let's do a loop of: Generate -> Local Search -> Update Best
        
        current_triples = _generate_random_linear_graph(n)
        current_alpha = _calc_alpha_ilp(current_triples, n)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = current_triples[:]
            
        if best_alpha <= 5:
            break
            
        # Local Search Phase
        no_improve_count = 0
        
        while time.time() < deadline:
            if best_alpha <= 5:
                break
                
            # Try to improve current_triples
            improved = False
            
            # Strategy 1: Swap a random triple with a random valid replacement
            if len(current_triples) > 0:
                idx = random.randint(0, len(current_triples) - 1)
                old_t = current_triples[idx]
                
                # Generate a few random candidate replacements
                # We can sample from all triples
                all_ts = list(itertools.combinations(range(n), 3))
                random.shuffle(all_ts)
                
                # Try up to 50 random candidates
                found_better = False
                best_candidate_alpha = current_alpha
                best_candidate_triples = None
                best_candidate_idx = idx
                best_candidate_t = None
                
                for new_t in all_ts[:50]:
                    # Check if new_t can replace old_t
                    # 1. Check linearity with the rest of the graph
                    rest = current_triples[:idx] + current_triples[idx+1:]
                    
                    # Quick linearity check for new_t against rest
                    a, b, c = new_t
                    pairs = [(a, b), (a, c), (b, c)]
                    
                    # Check against rest
                    valid = True
                    for rt in rest:
                        # Check if any pair in new_t is in rt? No, check shared points > 1
                        # Faster: check if any pair of new_t is a pair of rt?
                        # Actually, linearity means shared points <= 1.
                        # If shared points == 2, they share a pair.
                        # So we just need to ensure no pair from new_t is in any triple in rest.
                        # We can maintain a set of pairs for 'rest'? 
                        # For simplicity, just check intersection size.
                        if len(set(new_t) & set(rt)) > 1:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    test_triples = rest + [new_t]
                    new_alpha = _calc_alpha_ilp(test_triples, n)
                    
                    if new_alpha < best_candidate_alpha:
                        best_candidate_alpha = new_alpha
                        best_candidate_triples = test_triples
                        best_candidate_idx = idx
                        best_candidate_t = new_t
                        found_better = True
                        
                if found_better:
                    current_triples = best_candidate_triples
                    current_alpha = best_candidate_alpha
                    improved = True
                    no_improve_count = 0
                    
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = current_triples[:]
                        
            # Strategy 2: Add a triple (if possible)
            if not improved:
                # Try to add a triple to increase density
                all_ts = list(itertools.combinations(range(n), 3))
                random.shuffle(all_ts)
                
                added = False
                for new_t in all_ts[:20]:
                    # Check if new_t is linear with current_triples
                    valid = True
                    for ct in current_triples:
                        if len(set(new_t) & set(ct)) > 1:
                            valid = False
                            break
                    
                    if valid:
                        test_triples = current_triples + [new_t]
                        new_alpha = _calc_alpha_ilp(test_triples, n)
                        
                        # Adding a triple should not increase alpha, but might not decrease it either.
                        # We only accept if it decreases alpha (unlikely but possible) or if we want to explore.
                        # Actually, adding a triple *constrains* the independent set, so alpha can only stay same or decrease.
                        if new_alpha < current_alpha:
                            current_triples = test_triples
                            current_alpha = new_alpha
                            improved = True
                            no_improve_count = 0
                            
                            if current_alpha < best_alpha:
                                best_alpha = current_alpha
                                best_triples = current_triples[:]
                            added = True
                            break
                            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 30:
                    break # Restart local search with new random graph
                    
    return best_triples
# EVOLVE-BLOCK-END
