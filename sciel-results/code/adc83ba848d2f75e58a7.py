"""Improved program for dense increasing sequences with unique block products."""
import time
import itertools
import math

# EVOLVE-BLOCK-START

def _block_products(seq):
    """Return set of all nonempty consecutive block products for a sequence."""
    prods = set()
    n = len(seq)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            prods.add(p)
    return prods

def _is_valid(seq):
    """Check if sequence has all distinct block products."""
    return len(_block_products(seq)) == len(seq) * (len(seq) + 1) // 2

def _greedy_construct(candidates, deadline=None):
    """Greedy: add smallest candidate that keeps all block products distinct."""
    if deadline is None:
        deadline = time.time() + 58
    values = []
    products = set()
    for x in candidates:
        if time.time() > deadline:
            break
        # Compute new tail products
        tails = []
        prod = x
        tails.append(prod)
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values

def _beam_search_dynamic(candidates, deadline=None):
    """
    Beam search with dynamic beam width and potential-based heuristic.
    Uses wider beams early, narrows later. Scores beams by potential
    (how many more elements could potentially be added).
    """
    if deadline is None:
        deadline = time.time() + 58
    
    start_time = time.time()
    total_candidates = len(candidates)
    
    # Each beam: (sequence, set_of_products, last_index_in_candidates)
    # Start with empty sequence
    beams = [([], set(), 0)]
    best_seq = []
    
    # Dynamic beam width: start wide, narrow later
    # Phase 1 (first 40%): wide beam
    # Phase 2 (next 30%): medium beam  
    # Phase 3 (last 30%): narrow beam
    
    for step, x in enumerate(candidates):
        if time.time() > deadline:
            break
        
        # Determine beam width based on progress
        progress = step / total_candidates
        if progress < 0.4:
            beam_width = 80
        elif progress < 0.7:
            beam_width = 40
        else:
            beam_width = 20
        
        new_beams = []
        for seq, prods, _ in beams:
            # Try adding x
            tails = []
            prod = x
            tails.append(prod)
            for a in reversed(seq):
                prod *= a
                tails.append(prod)
            
            if not prods.intersection(tails):
                new_prods = prods | set(tails)
                new_beams.append((seq + [x], new_prods, step + 1))
        
        if not new_beams:
            break
        
        # Score beams by potential: prefer longer sequences, and sequences
        # that leave more room for future extensions
        def beam_score(beam):
            seq, prods, _ = beam
            # Primary: length
            # Secondary: potential = how many candidates > last element remain
            # that don't immediately conflict (approximate)
            if not seq:
                return (0, 0)
            last = seq[-1]
            remaining = sum(1 for c in candidates if c > last)
            # Potential heuristic: prefer sequences where the product set
            # is "sparse" - more room for future products
            # Approximate by: fewer products relative to possible range
            max_prod = max(prods) if prods else 1
            # Log of max product as a proxy for "space used"
            log_space = math.log(max_prod + 1) if max_prod > 1 else 0
            potential = remaining - log_space * 0.1
            return (len(seq), potential)
        
        new_beams.sort(key=beam_score, reverse=True)
        
        # Keep top beams
        if len(new_beams) > beam_width:
            beams = new_beams[:beam_width]
        else:
            beams = new_beams
        
        # Update best
        if beams[0][0] and len(beams[0][0]) > len(best_seq):
            best_seq = beams[0][0]
    
    return best_seq

def _beam_search_restart(candidates, deadline=None, max_restarts=3):
    """
    Beam search with restarts. Each restart uses a different subset of candidates
    or different ordering to explore diverse search paths.
    """
    if deadline is None:
        deadline = time.time() + 58
    
    best_overall = []
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Different strategies for each restart
        if restart == 0:
            # Standard order
            cands = list(candidates)
        elif restart == 1:
            # Skip some small numbers to explore different structures
            cands = [x for x in candidates if x >= 3]
        else:
            # Start from middle
            cands = [x for x in candidates if x >= 5]
        
        seq = _beam_search_dynamic(cands, deadline=deadline - 5)
        if len(seq) > len(best_overall):
            best_overall = seq
    
    return best_overall

def _local_search(seq, candidates, max_iter=200, deadline=None):
    """Try to extend or improve sequence by local modifications."""
    if deadline is None:
        deadline = time.time() + 58
    
    current = list(seq)
    current_prods = _block_products(current)
    best = list(current)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try extending: add a candidate at the end
        for x in candidates:
            if x <= current[-1] if current else False:
                continue
            if x in current:
                continue
            # Check if adding x is valid
            tails = []
            prod = x
            tails.append(prod)
            for a in reversed(current):
                prod *= a
                tails.append(prod)
            if not current_prods.intersection(tails):
                new_prods = current_prods | set(tails)
                new_seq = current + [x]
                if len(new_seq) > len(best):
                    best = new_seq
                current = new_seq
                current_prods = new_prods
                improved = True
                break
        
        if not improved:
            break
    
    return best

def construct():
    deadline = time.time() + 58  # Safety margin
    
    # Phase 1: Greedy construction (fast baseline)
    candidates = list(range(2, 129))
    greedy_seq = _greedy_construct(candidates, deadline)
    
    # Phase 2: Dynamic beam search with restarts
    if time.time() < deadline - 15:
        beam_seq = _beam_search_restart(candidates, deadline=deadline - 10, max_restarts=3)
        if len(beam_seq) > len(greedy_seq):
            greedy_seq = beam_seq
    
    # Phase 3: Local search / extension
    if time.time() < deadline - 5:
        # Try to extend the best sequence found
        remaining = [x for x in candidates if x not in set(greedy_seq)]
        extended = _local_search(greedy_seq, remaining, max_iter=100, deadline=deadline - 2)
        if len(extended) > len(greedy_seq):
            greedy_seq = extended
    
    # Final verification
    if _is_valid(greedy_seq):
        return greedy_seq
    else:
        # Fallback to greedy (should always be valid)
        return _greedy_construct(candidates, deadline=deadline)

# EVOLVE-BLOCK-END
