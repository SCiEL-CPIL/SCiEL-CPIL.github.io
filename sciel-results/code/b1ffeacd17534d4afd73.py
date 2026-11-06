"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _prime_exponent_vector(n):
    """Return tuple of exponents for primes 2,3,5,7,11,...,127 in n's factorization."""
    primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127]
    vec = []
    for p in primes:
        if p * p > n:
            break
        cnt = 0
        while n % p == 0:
            n //= p
            cnt += 1
        vec.append(cnt)
    if n > 1:
        vec.append(1)  # remaining prime factor
    return tuple(vec)

def _all_block_product_vectors(seq):
    """Return set of exponent vectors for all nonempty consecutive blocks."""
    n = len(seq)
    vecs = set()
    # Precompute prefix exponent vectors
    # For block [i..j], vector = prefix[j+1] - prefix[i]
    # We work with tuples; addition is component-wise
    prefix = [()]
    for x in seq:
        xv = _prime_exponent_vector(x)
        if not prefix:
            prefix.append(xv)
        else:
            pv = prefix[-1]
            xv_pad = xv + (0,) * (len(pv) - len(xv))
            pv_pad = pv + (0,) * (len(xv) - len(pv))
            prefix.append(tuple(a + b for a, b in zip(pv_pad, xv_pad)))
    
    for i in range(n):
        for j in range(i, n):
            pi = prefix[i]
            pj = prefix[j+1]
            # pad to same length
            maxlen = max(len(pi), len(pj))
            pi_pad = pi + (0,) * (maxlen - len(pi))
            pj_pad = pj + (0,) * (maxlen - len(pj))
            block_vec = tuple(b - a for a, b in zip(pi_pad, pj_pad))
            vecs.add(block_vec)
    return vecs

def _try_insert_at(seq, pos, x, existing_vecs=None):
    """Try to insert x at position pos in seq. Returns (success, new_vecs) or (False, None)."""
    # Compute all new block products that include x
    # Blocks ending at pos: [i..pos] for i in 0..pos
    # Blocks starting at pos: [pos..j] for j in pos..len(seq)
    # The singleton [pos..pos] is just x
    n = len(seq)
    xv = _prime_exponent_vector(x)
    maxlen = len(xv)
    
    # We need to check all blocks that include the new element at position pos
    # After insertion, seq becomes seq[:pos] + [x] + seq[pos:]
    # Blocks that include x:
    #   [i..pos] for i in 0..pos  (blocks ending at x)
    #   [pos..j] for j in pos..n-1 (blocks starting at x, going into old seq[pos:])
    # Note: [pos..pos] is just x, counted in both
    
    # Build prefix vectors for the new sequence
    new_seq = seq[:pos] + [x] + seq[pos:]
    new_n = len(new_seq)
    
    # Only need to check blocks that include index pos in new_seq
    # Compute prefix vectors for new_seq
    prefix = [()]
    for k, val in enumerate(new_seq):
        vv = _prime_exponent_vector(val)
        if not prefix:
            prefix.append(vv)
        else:
            pv = prefix[-1]
            vv_pad = vv + (0,) * (len(pv) - len(vv))
            pv_pad = pv + (0,) * (len(vv) - len(pv))
            prefix.append(tuple(a + b for a, b in zip(pv_pad, vv_pad)))
    
    # Check blocks including position pos
    for i in range(pos + 1):  # blocks [i..pos]
        pi = prefix[i]
        pp = prefix[pos + 1]
        maxlen = max(len(pi), len(pp))
        pi_pad = pi + (0,) * (maxlen - len(pi))
        pp_pad = pp + (0,) * (maxlen - len(pp))
        block_vec = tuple(b - a for a, b in zip(pi_pad, pp_pad))
        if existing_vecs is not None and block_vec in existing_vecs:
            return False, None
    
    for j in range(pos, new_n):  # blocks [pos..j]
        pp = prefix[pos]
        pj = prefix[j + 1]
        maxlen = max(len(pp), len(pj))
        pp_pad = pp + (0,) * (maxlen - len(pp))
        pj_pad = pj + (0,) * (maxlen - len(pj))
        block_vec = tuple(b - a for a, b in zip(pp_pad, pj_pad))
        if existing_vecs is not None and block_vec in existing_vecs:
            return False, None
    
    # Also need to verify the new blocks don't collide with each other
    # (but they're all new, so just need to check they're distinct among themselves)
    new_block_vecs = set()
    for i in range(pos + 1):
        pi = prefix[i]
        pp = prefix[pos + 1]
        maxlen = max(len(pi), len(pp))
        pi_pad = pi + (0,) * (maxlen - len(pi))
        pp_pad = pp + (0,) * (maxlen - len(pp))
        block_vec = tuple(b - a for a, b in zip(pi_pad, pp_pad))
        if block_vec in new_block_vecs:
            return False, None
        new_block_vecs.add(block_vec)
    
    for j in range(pos, new_n):
        pp = prefix[pos]
        pj = prefix[j + 1]
        maxlen = max(len(pp), len(pj))
        pp_pad = pp + (0,) * (maxlen - len(pp))
        pj_pad = pj + (0,) * (maxlen - len(pj))
        block_vec = tuple(b - a for a, b in zip(pp_pad, pj_pad))
        if block_vec in new_block_vecs:
            return False, None
        new_block_vecs.add(block_vec)
    
    return True, new_block_vecs

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    # Step 1: Start with all primes in [2, 128]
    primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127]
    seq = list(primes)
    
    # Compute all block product vectors for initial sequence
    existing_vecs = _all_block_product_vectors(seq)
    
    # Step 2: Try to insert composites in randomized order
    composites = [x for x in range(2, 129) if x not in primes]
    random.seed(42)
    
    # Multiple passes with randomization
    best_len = len(seq)
    best_seq = list(seq)
    
    for trial in range(50):
        if time.time() > deadline - 5.0:
            break
        
        seq = list(primes)
        existing_vecs = _all_block_product_vectors(seq)
        
        shuffled = list(composites)
        random.shuffle(shuffled)
        
        for x in shuffled:
            if time.time() > deadline - 2.0:
                break
            
            # Try all positions
            found = False
            for pos in range(len(seq) + 1):
                # Quick check: x must fit in sorted order or we allow non-sorted? 
                # The problem says strictly increasing, so we need to maintain sorted order
                # Check if inserting at pos maintains sorted order
                if pos > 0 and seq[pos-1] >= x:
                    continue
                if pos < len(seq) and seq[pos] <= x:
                    continue
                
                success, new_vecs = _try_insert_at(seq, pos, x, existing_vecs)
                if success:
                    seq.insert(pos, x)
                    existing_vecs = _all_block_product_vectors(seq)
                    found = True
                    break
            
            if not found:
                continue
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)
    
    # Step 3: Local search - try swapping or removing/adding
    if time.time() < deadline - 1.0:
        # Try adding back some composites that were excluded
        for x in composites:
            if time.time() > deadline - 1.0:
                break
            if x in best_seq:
                continue
            for pos in range(len(best_seq) + 1):
                if pos > 0 and best_seq[pos-1] >= x:
                    continue
                if pos < len(best_seq) and best_seq[pos] <= x:
                    continue
                success, new_vecs = _try_insert_at(best_seq, pos, x, _all_block_product_vectors(best_seq))
                if success:
                    best_seq.insert(pos, x)
                    break
    
    return best_seq
# EVOLVE-BLOCK-END
