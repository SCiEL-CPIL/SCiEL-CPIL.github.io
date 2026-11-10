"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import permutations

def _find_overlap_suffix_prefix(new_tail, block_prefix):
    """Find the longest suffix of new_tail that equals a prefix of block_prefix."""
    max_ov = min(len(new_tail), len(block_prefix))
    for o in range(max_ov, 0, -1):
        if new_tail[-o:] == block_prefix[:o]:
            return o
    return 0

def _build_word_from_order(order, m):
    """Build a superpermutation candidate from a given permutation order."""
    new = ""
    for p in order:
        block = p + str(m) + p
        tail = new[-len(block):] if len(new) >= len(block) else new
        ov = _find_overlap_suffix_prefix(tail, block)
        new += block[ov:]
    return new

def _verify_superpermutation(word, m):
    """Check if word contains all permutations of 1..m as consecutive symbols."""
    if len(word) < m:
        return False
    all_perms = set(permutations(range(1, m + 1)))
    found = set()
    for i in range(len(word) - m + 1):
        w = tuple(int(c) for c in word[i:i + m])
        if len(set(w)) == m:
            found.add(w)
    return found == all_perms

def _soft_max_order(perms, m, temperature, rng):
    """Generate a permutation order using soft-max stochastic selection."""
    remaining = list(perms)
    order = []
    current_tail = ""
    
    while remaining:
        if not remaining:
            break
        
        # Score each candidate by overlap with current tail
        scores = []
        for p in remaining:
            block = p + str(m) + p
            tail = current_tail[-len(block):] if len(current_tail) >= len(block) else current_tail
            ov = _find_overlap_suffix_prefix(tail, block)
            # Higher overlap is better; add small noise for exploration
            score = ov + rng.gauss(0, temperature)
            scores.append(score)
        
        # Soft-max selection
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]
        
        # Sample from distribution
        r = rng.random()
        cumulative = 0
        chosen_idx = 0
        for i, prob in enumerate(probs):
            cumulative += prob
            if r <= cumulative:
                chosen_idx = i
                break
        
        chosen = remaining.pop(chosen_idx)
        order.append(chosen)
        
        # Update current_tail
        block = chosen + str(m) + chosen
        tail = current_tail[-len(block):] if len(current_tail) >= len(block) else current_tail
        ov = _find_overlap_suffix_prefix(tail, block)
        current_tail = (current_tail + block[ov:])[-(2*m + 1):]
    
    return order

def construct(n):
    start_time = time.time()
    deadline = start_time + 115
    
    word = "1"
    for m in range(2, n + 1):
        if time.time() > deadline:
            break

        # Extract all distinct (m-1)-permutations from word
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = word[i:i + m - 1]
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)

        if m < n:
            new = ""
            for p in perms:
                block = p + str(m) + p
                tail = new[-len(block):] if len(new) >= len(block) else new
                ov = _find_overlap_suffix_prefix(tail, block)
                new += block[ov:]
            word = new
        else:
            best_word, best_len = None, float('inf')
            
            # Try deterministic candidates first
            candidates = [perms, list(reversed(perms)), sorted(perms)]
            for ordering in candidates:
                if time.time() > deadline - 30:
                    break
                new = _build_word_from_order(ordering, m)
                if len(new) < m:
                    continue
                if _verify_superpermutation(new, m) and len(new) < best_len:
                    best_len, best_word = len(new), new
            
            # Stochastic restarts with soft-max
            if time.time() < deadline - 20:
                rng = random.Random(42)
                num_restarts = 50
                base_temp = 0.5
                for restart in range(num_restarts):
                    if time.time() > deadline - 10:
                        break
                    # Anneal temperature
                    temp = base_temp * (0.95 ** restart)
                    order = _soft_max_order(perms, m, temp, rng)
                    new = _build_word_from_order(order, m)
                    if len(new) < m:
                        continue
                    if _verify_superpermutation(new, m) and len(new) < best_len:
                        best_len, best_word = len(new), new
            
            if best_word is not None:
                word = best_word
            else:
                new = ""
                for p in perms:
                    block = p + str(m) + p
                    tail = new[-len(block):] if len(new) >= len(block) else new
                    ov = _find_overlap_suffix_prefix(tail, block)
                    new += block[ov:]
                word = new

    return word
# EVOLVE-BLOCK-END
