"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _reed_muller_seed():
    """First-order Reed-Muller code on 16 coordinates, extended by two zero bits."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def _greedy_build(seed_words, deadline, rng, max_candidates=200000):
    """Greedily add random candidates to seed_words maintaining min distance 6."""
    code = list(seed_words)
    code_set = set(code)
    min_dist = 6
    
    # Precompute: for each word in code, we need to check distance >= 6
    # Use a list for fast iteration
    
    start_time = time.time()
    candidates_checked = 0
    
    # First pass: try to add from a large pool of random candidates
    # We'll generate random 18-bit integers and check them
    # To be efficient, we check against existing code words
    
    # Sort code by weight for potential early termination (optional optimization)
    # For now, just iterate through all code words
    
    while time.time() < deadline:
        # Generate a batch of random candidates
        batch_size = 5000
        candidates = [rng.randint(0, 2**18 - 1) for _ in range(batch_size)]
        
        for cand in candidates:
            if time.time() >= deadline:
                break
            if cand in code_set:
                continue
            candidates_checked += 1
            
            # Check distance against all existing code words
            ok = True
            for w in code:
                if (w ^ cand).bit_count() < min_dist:
                    ok = False
                    break
            
            if ok:
                code.append(cand)
                code_set.add(cand)
    
    return code

def _greedy_with_priority(seed_words, deadline, rng):
    """Greedy build with priority on high-weight words first (heuristic)."""
    code = list(seed_words)
    code_set = set(code)
    min_dist = 6
    
    # Generate a pool of candidates, sorted by weight (higher weight might be better)
    # Actually, let's just do random but with more aggressive checking
    
    start_time = time.time()
    
    # Try adding candidates in a structured way
    # Generate all 2^18 words? No, that's 262144, which is feasible for checking
    # But checking each against a growing code is O(n^2)
    
    # Instead, let's do multiple passes
    for iteration in range(50):
        if time.time() >= deadline:
            break
        
        # Generate random candidates
        batch_size = 10000
        for _ in range(batch_size):
            if time.time() >= deadline:
                break
            cand = rng.randint(0, 2**18 - 1)
            if cand in code_set:
                continue
            
            ok = True
            for w in code:
                if (w ^ cand).bit_count() < min_dist:
                    ok = False
                    break
            
            if ok:
                code.append(cand)
                code_set.add(cand)
    
    return code

def construct():
    deadline = time.time() + 85  # Safety margin: 5s before 90s limit
    
    # Start with Reed-Muller seed
    seed_words = _reed_muller_seed()
    
    # Try multiple random restarts with different seeds
    best_code = seed_words
    
    # First, try a simple greedy with a fixed seed for reproducibility
    rng = random.Random(42)
    code1 = _greedy_build(seed_words, deadline, rng)
    if len(code1) > len(best_code):
        best_code = code1
    
    # Try different random seeds for diversity
    for seed in [123, 456, 789, 1000, 2024]:
        if time.time() >= deadline:
            break
        rng = random.Random(seed)
        code = _greedy_build(seed_words, deadline, rng)
        if len(code) > len(best_code):
            best_code = code
    
    return best_code
# EVOLVE-BLOCK-END
