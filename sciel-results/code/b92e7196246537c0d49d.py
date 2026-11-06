"""Stronger algorithm for dense increasing sequences with unique block products."""
import time
import random
from functools import lru_cache

# EVOLVE-BLOCK-START

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _all_block_products(seq):
    """Return the set of all nonempty consecutive block products."""
    prods = set()
    n = len(seq)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            prods.add(p)
    return prods


def _validate(seq):
    """Check if seq is strictly increasing in [2,128] and all block products distinct."""
    if not seq:
        return True
    if seq[0] < 2 or seq[-1] > 128:
        return False
    for i in range(1, len(seq)):
        if seq[i] <= seq[i - 1]:
            return False
    prods = _all_block_products(seq)
    return len(prods) == len(seq) * (len(seq) + 1) // 2


def _greedy_forward():
    """Greedy: try each x in [2,128], add if all new tail products are fresh."""
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values


def _greedy_prime_first():
    """Greedy but prefer primes first, then composites."""
    primes = [x for x in range(2, 129) if _is_prime(x)]
    composites = [x for x in range(2, 129) if not _is_prime(x)]
    values, products = [], set()
    for x in primes + composites:
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values


def _insert_composites(seq):
    """Try to insert composites into a prime-based sequence at valid positions."""
    seq = list(seq)
    composites = [x for x in range(2, 129) if not _is_prime(x)]
    for c in composites:
        if c in seq:
            continue
        inserted = False
        for i in range(len(seq) + 1):
            if i > 0 and seq[i - 1] >= c:
                continue
            if i < len(seq) and seq[i] <= c:
                continue
            new_seq = seq[:i] + [c] + seq[i:]
            if _validate(new_seq):
                seq = new_seq
                inserted = True
                break
    return seq


def _local_refine(seq, deadline):
    """Local search: try swapping, inserting, removing to improve length."""
    seq = list(seq)
    best_len = len(seq)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try inserting each unused number at each valid position
        used = set(seq)
        for c in range(2, 129):
            if c in used:
                continue
            for i in range(len(seq) + 1):
                if i > 0 and seq[i - 1] >= c:
                    continue
                if i < len(seq) and seq[i] <= c:
                    continue
                new_seq = seq[:i] + [c] + seq[i:]
                if _validate(new_seq):
                    seq = new_seq
                    best_len = len(seq)
                    improved = True
                    break
            if time.time() > deadline:
                break
    return seq


def _beam_search(deadline, beam_width=200, max_depth=25):
    """Beam search over sequence prefixes, tracking products incrementally."""
    # Each beam entry: (seq_tuple, products_set)
    # Start with empty
    beam = [((), frozenset())]
    best_seq = []
    best_len = 0
    
    for depth in range(max_depth):
        if time.time() > deadline:
            break
        new_beam = []
        for seq, prods in beam:
            if time.time() > deadline:
                break
            # Try adding each x > last element
            last = seq[-1] if seq else 1
            for x in range(last + 1, 129):
                # Compute new tail products
                new_prods_list = [x]
                prod = x
                for a in reversed(seq):
                    prod *= a
                    new_prods_list.append(prod)
                new_prods_set = set(new_prods_list)
                if not new_prods_set & set(prods):
                    new_seq = seq + (x,)
                    new_beam.append((new_seq, frozenset(prods | new_prods_set)))
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = list(new_seq)
        
        if not new_beam:
            break
        
        # Sort by length descending, then lexicographically, keep top beam_width
        new_beam.sort(key=lambda e: (-len(e[0]), e[0]))
        beam = new_beam[:beam_width]
        
        if len(beam) < beam_width:
            # No more extensions possible
            break
    
    return best_seq


def _beam_search_randomized(deadline, beam_width=100, max_depth=25, seed=42):
    """Beam search with randomized ordering to escape local optima."""
    rng = random.Random(seed)
    beam = [((), frozenset())]
    best_seq = []
    best_len = 0
    
    for depth in range(max_depth):
        if time.time() > deadline:
            break
        new_beam = []
        for seq, prods in beam:
            if time.time() > deadline:
                break
            last = seq[-1] if seq else 1
            candidates = list(range(last + 1, 129))
            rng.shuffle(candidates)
            for x in candidates:
                new_prods_list = [x]
                prod = x
                for a in reversed(seq):
                    prod *= a
                    new_prods_list.append(prod)
                new_prods_set = set(new_prods_list)
                if not new_prods_set & set(prods):
                    new_seq = seq + (x,)
                    new_beam.append((new_seq, frozenset(prods | new_prods_set)))
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = list(new_seq)
        
        if not new_beam:
            break
        
        new_beam.sort(key=lambda e: (-len(e[0]), e[0]))
        beam = new_beam[:beam_width]
        
        if len(beam) < beam_width:
            break
    
    return best_seq


def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    candidates = []
    
    # Strategy 1: Simple greedy
    g1 = _greedy_forward()
    candidates.append(g1)
    
    # Strategy 2: Prime-first greedy
    g2 = _greedy_prime_first()
    candidates.append(g2)
    
    # Strategy 3: Prime-based + composite insertion
    primes = [x for x in range(2, 129) if _is_prime(x)]
    seq3 = _insert_composites(primes)
    candidates.append(seq3)
    
    # Strategy 4: Beam search (deterministic)
    if time.time() < deadline - 5:
        bs1 = _beam_search(deadline - 5, beam_width=150, max_depth=20)
        if len(bs1) > 0:
            candidates.append(bs1)
    
    # Strategy 5: Randomized beam search
    if time.time() < deadline - 5:
        bs2 = _beam_search_randomized(deadline - 5, beam_width=100, max_depth=20, seed=123)
        if len(bs2) > 0:
            candidates.append(bs2)
    
    # Strategy 6: Local refinement on best candidate
    if time.time() < deadline - 3:
        best_so_far = max(candidates, key=len)
        refined = _local_refine(best_so_far, deadline - 3)
        candidates.append(refined)
    
    # Strategy 7: Beam search from best local result
    if time.time() < deadline - 3:
        best_so_far = max(candidates, key=len)
        # Try extending with beam from the current best
        remaining = deadline - 3
        bs3 = _beam_search(remaining, beam_width=80, max_depth=15)
        if len(bs3) > 0:
            candidates.append(bs3)
    
    # Pick the longest valid result
    best = max(candidates, key=len)
    
    # Final validation
    if not _validate(best):
        best = _greedy_forward()
    
    return best

# EVOLVE-BLOCK-END
