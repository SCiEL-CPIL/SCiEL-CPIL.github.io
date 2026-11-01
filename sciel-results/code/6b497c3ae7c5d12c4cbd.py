"""ILP-based covering design C(24,7,3) with greedy initialization."""
import itertools
import random
import time
import sys

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def _precompute_triples(v, t):
    """Precompute all t-subsets and map each to an index."""
    triples = list(itertools.combinations(range(v), t))
    triple_idx = {ts: i for i, ts in enumerate(triples)}
    return triples, triple_idx, len(triples)

def _greedy_init(v, k, t, triples, triple_idx, rng, time_limit=5.0):
    """Fast greedy to get a starting covering."""
    deadline = time.time() + time_limit
    uncovered = set(range(len(triples)))
    blocks = []
    
    while uncovered and time.time() < deadline:
        # Pick random uncovered triple
        ts_idx = rng.choice(list(uncovered))
        ts = triples[ts_idx]
        block = list(ts)
        
        while len(block) < k:
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
            best_x = None
            best_gain = -1
            for x in candidates:
                gain = 0
                for S in itertools.combinations(block, t - 1):
                    if x not in S:
                        tset = tuple(sorted(S + (x,)))
                        if triple_idx[tset] in uncovered:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        
        if len(block) < k:
            break
        block.sort()
        blocks.append(tuple(block))
        
        covered = set(triple_idx[ts] for ts in itertools.combinations(block, t))
        uncovered -= covered
    
    return [list(b) for b in blocks]

def _remove_redundant(blocks, triples, triple_idx, v, k, t, time_limit=5.0):
    """Remove redundant blocks to reduce population size."""
    deadline = time.time() + time_limit
    blocks = [list(b) for b in blocks]
    
    while time.time() < deadline:
        n = len(blocks)
        if n == 0:
            break
        
        # Build coverage counts
        n_triples = len(triples)
        counts = np.zeros(n_triples, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                counts[triple_idx[ts]] += 1
        
        # Find redundant block
        redundant_idx = -1
        for i in range(n):
            b = blocks[i]
            is_redundant = True
            for ts in itertools.combinations(b, t):
                idx = triple_idx[ts]
                if counts[idx] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                redundant_idx = i
                break
        
        if redundant_idx == -1:
            break
        
        # Remove it
        blocks.pop(redundant_idx)
    
    return blocks

def _is_valid_covering(blocks, v, k, t, triples, triple_idx):
    """Check validity."""
    n_triples = len(triples)
    counts = np.zeros(n_triples, dtype=np.int32)
    for b in blocks:
        for ts in itertools.combinations(b, t):
            counts[triple_idx[ts]] += 1
    return int(np.sum(counts == 0)) == 0

def _ilp_refine(v, k, t, triples, triple_idx, initial_blocks, rng, time_limit):
    """
    Use ILP to refine the covering by selecting a subset of candidate blocks.
    Candidate blocks include initial blocks plus some random perturbations.
    """
    deadline = time.time() + time_limit
    n_triples = len(triples)
    
    # Generate candidate blocks: initial + random perturbations
    candidates = [tuple(sorted(b)) for b in initial_blocks]
    
    # Add random blocks
    num_random = 50
    for _ in range(num_random):
        rb = tuple(sorted(rng.sample(range(v), k)))
        if rb not in candidates:
            candidates.append(rb)
    
    # Remove duplicates
    candidates = list(set(candidates))
    n_candidates = len(candidates)
    
    # Build incidence matrix: A[i, j] = 1 if triple i is in candidate block j
    A = np.zeros((n_triples, n_candidates), dtype=np.int8)
    for j, block in enumerate(candidates):
        for ts in itertools.combinations(block, t):
            idx = triple_idx[ts]
            A[idx, j] = 1
    
    # Objective: minimize sum of x_j
    c = np.ones(n_candidates, dtype=np.float64)
    
    # Constraints: A @ x >= 1 for all triples
    lb = np.ones(n_triples, dtype=np.float64)
    ub = np.full(n_triples, np.inf, dtype=np.float64)
    constraints = LinearConstraint(A, lb, ub)
    
    # Bounds: 0 <= x_j <= 1
    lb_x = np.zeros(n_candidates, dtype=np.float64)
    ub_x = np.ones(n_candidates, dtype=np.float64)
    bounds = Bounds(lb_x, ub_x)
    
    # Integer constraints
    integrality = np.ones(n_candidates, dtype=np.int8)
    
    # Solve ILP
    try:
        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if result.success and result.x is not None:
            # Extract selected blocks
            selected = []
            for j in range(n_candidates):
                if result.x[j] > 0.5:
                    selected.append(list(candidates[j]))
            
            # Validate
            if _is_valid_covering(selected, v, k, t, triples, triple_idx):
                return selected
            
            # If ILP solution is invalid, fall back to initial blocks
            return initial_blocks
        else:
            return initial_blocks
    except Exception:
        return initial_blocks

def construct(v, k, t):
    start_time = time.time()
    total_budget = 140.0  # Safety margin for 150s kill
    
    # Precompute
    triples, triple_idx, n_triples = _precompute_triples(v, t)
    
    rng = random.Random(42)
    
    # Phase 1: Multiple greedy initializations
    best_blocks = None
    best_size = float('inf')
    
    num_greedy = 5
    per_greedy = 3.0
    for i in range(num_greedy):
        if time.time() - start_time > total_budget - 30.0:
            break
        gb = _greedy_init(v, k, t, triples, triple_idx, rng, time_limit=per_greedy)
        if len(gb) < best_size:
            best_size = len(gb)
            best_blocks = gb
    
    if best_blocks is None:
        best_blocks = _greedy_init(v, k, t, triples, triple_idx, rng, 5.0)
    
    # Phase 2: Remove redundant blocks
    remaining = total_budget - (time.time() - start_time) - 20.0
    if remaining > 2.0:
        best_blocks = _remove_redundant(best_blocks, triples, triple_idx, v, k, t, time_limit=min(5.0, remaining))
    
    # Phase 3: ILP refinement
    remaining = total_budget - (time.time() - start_time) - 5.0
    if remaining > 10.0:
        ilp_blocks = _ilp_refine(v, k, t, triples, triple_idx, best_blocks, rng, remaining)
        if ilp_blocks and len(ilp_blocks) <= len(best_blocks):
            best_blocks = ilp_blocks
        # Remove redundant again
        remaining2 = total_budget - (time.time() - start_time) - 3.0
        if remaining2 > 1.0:
            best_blocks = _remove_redundant(best_blocks, triples, triple_idx, v, k, t, time_limit=min(3.0, remaining2))
    
    # Ensure validity
    if not _is_valid_covering(best_blocks, v, k, t, triples, triple_idx):
        # Fallback to greedy
        best_blocks = _greedy_init(v, k, t, triples, triple_idx, rng, 10.0)
    
    return best_blocks
