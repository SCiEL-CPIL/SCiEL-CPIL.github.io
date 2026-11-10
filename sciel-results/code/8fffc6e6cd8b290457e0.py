"""Construct the largest admissible tuple within [0, 5000]."""
import time
import random
import numpy as np

N = 5000
DEADLINE = 325.0  # safety margin before 330s kill

def _sieve_primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i, v in enumerate(sieve) if v]

def _is_admissible(points, primes):
    """Check if points form an admissible tuple."""
    n = len(points)
    if n == 0:
        return True
    arr = np.array(points)
    for p in primes:
        if p > n:
            break
        residues = arr % p
        if len(np.unique(residues)) == p:
            return False
    return True

def _greedy_construct(primes, rng, start_time, max_time, seed_points=None):
    """
    Greedily construct an admissible tuple.
    Uses vectorized residue tracking for fast admissibility checks.
    """
    if seed_points is None:
        # Start with a reasonable seed
        seed_points = [1] + [p for p in primes if p * p > N and p <= N]
    
    point_set = set(seed_points)
    points = list(seed_points)
    
    # Track residue coverage: for each prime p <= current_len, track which residues are covered
    # Use a dict: p -> set of covered residues
    max_track = 200  # track primes up to 200 for efficiency
    
    def _get_residue_state(pts):
        state = {}
        n = len(pts)
        arr = np.array(pts) if pts else np.array([])
        for p in primes:
            if p > n:
                break
            residues = arr % p
            state[p] = set(residues.tolist())
        return state
    
    residue_state = _get_residue_state(points)
    
    # Precompute candidates sorted by a heuristic score
    # Prefer values that are "rare" in residue patterns for small primes
    candidates = [v for v in range(N + 1) if v not in point_set]
    
    # Score: penalize values that complete residue classes for small primes
    def _candidate_score(v):
        score = 0.0
        for p in primes:
            if p > len(points) + 1:
                break
            if p in residue_state:
                covered = residue_state[p]
                if len(covered) == p - 1 and v % p not in covered:
                    score += 100 * p  # would complete this prime's residues
                elif v % p in covered:
                    score += 1  # already covered, less useful
        return score
    
    # Sort candidates by score (lower is better)
    # For efficiency, only sort a subset
    if len(candidates) > 5000:
        # Sample and sort
        sample = rng.sample(candidates, min(5000, len(candidates)))
        sample.sort(key=_candidate_score)
        rest = [v for v in candidates if v not in set(sample)]
        candidates = sample + rest
    else:
        candidates.sort(key=_candidate_score)
    
    # Greedily add candidates
    added_count = 0
    for v in candidates:
        if time.time() - start_time > max_time - 5:
            break
        if v in point_set:
            continue
        
        # Check if adding v maintains admissibility
        new_n = len(points) + 1
        valid = True
        
        for p in primes:
            if p > new_n:
                break
            if p in residue_state:
                covered = residue_state[p]
            else:
                # Need to compute for this prime
                arr = np.array(points)
                covered = set((arr % p).tolist())
                residue_state[p] = covered
            
            if len(covered) == p:
                valid = False
                break
            elif len(covered) == p - 1 and v % p not in covered:
                # Adding v would complete all p residue classes
                valid = False
                break
        
        if valid:
            points.append(v)
            point_set.add(v)
            added_count += 1
            
            # Update residue state
            for p in primes:
                if p <= new_n:
                    if p not in residue_state:
                        arr = np.array(points)
                        residue_state[p] = set((arr % p).tolist())
                    else:
                        residue_state[p].add(v % p)
    
    return points

def _local_search(points, primes, rng, start_time, max_time, max_iterations=10000):
    """
    Local search: try to add points, and if stuck, try swaps.
    """
    point_set = set(points)
    best = list(points)
    best_len = len(best)
    
    def _get_residue_state(pts):
        state = {}
        n = len(pts)
        arr = np.array(pts) if pts else np.array([])
        for p in primes:
            if p > n:
                break
            residues = arr % p
            state[p] = set(residues.tolist())
        return state
    
    residue_state = _get_residue_state(points)
    
    def _try_add(v):
        nonlocal residue_state
        if v in point_set:
            return False
        new_n = len(points) + 1
        for p in primes:
            if p > new_n:
                break
            if p in residue_state:
                covered = residue_state[p]
            else:
                arr = np.array(points)
                covered = set((arr % p).tolist())
                residue_state[p] = covered
            if len(covered) == p:
                return False
            elif len(covered) == p - 1 and v % p not in covered:
                return False
        # Add v
        points.append(v)
        point_set.add(v)
        for p in primes:
            if p <= new_n:
                if p not in residue_state:
                    arr = np.array(points)
                    residue_state[p] = set((arr % p).tolist())
                else:
                    residue_state[p].add(v % p)
        return True
    
    def _remove(v):
        nonlocal residue_state
        points.remove(v)
        point_set.discard(v)
        new_n = len(points)
        # Update residue state
        for p in primes:
            if p > new_n:
                residue_state.pop(p, None)
                continue
            if p in residue_state:
                # Recompute if needed (simpler than tracking counts)
                if len(points) < 100:
                    arr = np.array(points)
                    residue_state[p] = set((arr % p).tolist())
                else:
                    # Keep as is for speed; will be recomputed if needed
                    pass
    
    iteration = 0
    stuck_counter = 0
    
    while iteration < max_iterations and time.time() - start_time < max_time - 10:
        iteration += 1
        
        # Try to add a random point
        v = rng.randint(0, N)
        if _try_add(v):
            if len(points) > best_len:
                best = list(points)
                best_len = len(best)
            stuck_counter = 0
        else:
            stuck_counter += 1
        
        # If stuck for a while, try a swap
        if stuck_counter > 50 and len(points) > 10:
            stuck_counter = 0
            # Try to swap: remove one point, add another
            # Pick a point to remove that's "least useful"
            remove_candidates = list(point_set)
            rng.shuffle(remove_candidates)
            
            for v_remove in remove_candidates[:5]:
                # Tentatively remove
                points.remove(v_remove)
                point_set.discard(v_remove)
                
                # Try to add a new point
                added = False
                for _ in range(20):
                    v_add = rng.randint(0, N)
                    if v_add in point_set:
                        continue
                    new_n = len(points) + 1
                    valid = True
                    for p in primes:
                        if p > new_n:
                            break
                        if p in residue_state:
                            covered = residue_state[p]
                        else:
                            arr = np.array(points)
                            covered = set((arr % p).tolist())
                            residue_state[p] = covered
                        if len(covered) == p:
                            valid = False
                            break
                        elif len(covered) == p - 1 and v_add % p not in covered:
                            valid = False
                            break
                    
                    if valid:
                        points.append(v_add)
                        point_set.add(v_add)
                        for p in primes:
                            if p <= new_n:
                                if p not in residue_state:
                                    arr = np.array(points)
                                    residue_state[p] = set((arr % p).tolist())
                                else:
                                    residue_state[p].add(v_add % p)
                        added = True
                        break
                
                if added and len(points) >= best_len:
                    if len(points) > best_len:
                        best = list(points)
                        best_len = len(best)
                    break
                else:
                    # Restore
                    points.append(v_remove)
                    point_set.add(v_remove)
                    # Recompute residue state
                    residue_state = _get_residue_state(points)
    
    return best

def _repair(points, primes):
    """
    Repair: ensure admissibility by removing points if needed.
    """
    point_set = set(points)
    
    n = len(points)
    for p in primes:
        if p > n:
            break
        arr = np.array(points)
        residues = arr % p
        if len(np.unique(residues)) == p:
            # Need to remove at least one point from each residue class
            # Remove from the least-populated class
            counts = np.bincount(residues, minlength=p)
            min_class = np.argmin(counts)
            # Find a point in this class and remove it
            for v in points:
                if v % p == min_class:
                    points.remove(v)
                    point_set.discard(v)
                    break
            # Recursively repair
            return _repair(points, primes)
    
    return points

def construct():
    start_time = time.time()
    max_time = DEADLINE
    
    primes = _sieve_primes(N + 1)
    rng = random.Random(42)
    
    best_points = []
    best_len = 0
    
    # Multiple restarts with different seeds
    num_restarts = 8
    time_per_restart = (max_time - 30) / num_restarts  # leave 30s for final polish
    
    for restart in range(num_restarts):
        if time.time() - start_time > max_time - 30:
            break
        
        # Different seed for each restart
        seed_rng = random.Random(42 + restart * 1000)
        
        # Create initial seed points
        if restart == 0:
            seed_points = [1] + [p for p in primes if p * p > N and p <= N]
        elif restart == 1:
            seed_points = list(range(0, 50, 7))  # arithmetic progression
        elif restart == 2:
            seed_points = [v for v in range(N + 1) if v % 6 in (1, 5)][:200]
        else:
            # Random subset
            seed_points = seed_rng.sample(range(N + 1), 100)
        
        # Greedy construction
        points = _greedy_construct(primes, seed_rng, start_time, start_time + time_per_restart, seed_points)
        
        # Local search
        points = _local_search(points, primes, seed_rng, start_time, start_time + time_per_restart)
        
        # Repair
        points = _repair(points, primes)
        
        # Verify
        if _is_admissible(points, primes):
            if len(points) > best_len:
                best_points = list(points)
                best_len = len(points)
        else:
            # Fallback: repair more aggressively
            points = _repair(points, primes)
            if _is_admissible(points, primes) and len(points) > best_len:
                best_points = list(points)
                best_len = len(points)
    
    # Final polish: try to add more points to the best solution
    if best_points and time.time() - start_time < max_time - 5:
        best_points = _local_search(best_points, primes, rng, start_time, max_time, max_iterations=5000)
        best_points = _repair(best_points, primes)
        if not _is_admissible(best_points, primes):
            # Fallback to previous best
            best_points = [1] + [p for p in primes if p * p > N and p <= N]
    
    # Final verification
    if not _is_admissible(best_points, primes):
        best_points = [1] + [p for p in primes if p * p > N and p <= N]
    
    return best_points
