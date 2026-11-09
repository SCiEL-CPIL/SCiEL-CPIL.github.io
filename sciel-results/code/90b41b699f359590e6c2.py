"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

N = 18
MIN_DISTANCE = 6
DEADLINE = 85  # seconds, safety margin before 90s kill
TOTAL_WORDS = 1 << N  # 262144

def _reed_muller_base():
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

def _precompute_neighbors():
    """
    Precompute for each word, the list of all words within Hamming distance < MIN_DISTANCE.
    Actually, we'll precompute the set of words at each distance from 0, then use XOR.
    For a given selected word s, the forbidden words are {s ^ v : v in ball(0, r)} where r = MIN_DISTANCE - 1.
    """
    # Precompute all vectors of weight <= MIN_DISTANCE - 1
    # These are the offsets from a codeword that are "too close"
    offsets_by_weight = [[] for _ in range(MIN_DISTANCE)]  # weights 0 to MIN_DISTANCE-1
    for w in range(TOTAL_WORDS):
        wt = w.bit_count()
        if wt < MIN_DISTANCE:
            offsets_by_weight[wt].append(w)
    
    # Combine into a flat list of all offsets with weight < MIN_DISTANCE
    all_offsets = []
    for wt in range(MIN_DISTANCE):
        all_offsets.extend(offsets_by_weight[wt])
    
    return all_offsets, offsets_by_weight

def _greedy_bitset(deadline, seed=0, order='weight_asc'):
    """
    Bitset-based greedy: maintain a boolean array 'forbidden' of size 2^N.
    When we add a codeword s, mark all s^offset as forbidden for offset in all_offsets.
    Then scan for available words in the desired order.
    """
    random.seed(seed)
    start_time = time.time()
    
    # Precompute offsets
    all_offsets, offsets_by_weight = _precompute_neighbors()
    
    # Forbidden bitset: use a bytearray for memory efficiency
    # forbidden[i] = 1 means word i is too close to some selected word
    forbidden = bytearray(TOTAL_WORDS)
    
    selected = []
    
    # Start with word 0
    selected.append(0)
    forbidden[0] = 1
    for off in all_offsets:
        forbidden[off] = 1  # 0 ^ off = off
    
    # Build candidate lists based on order
    if order == 'weight_asc':
        words_by_weight = [[] for _ in range(N + 1)]
        for w in range(1, TOTAL_WORDS):
            wt = w.bit_count()
            words_by_weight[wt].append(w)
        for wt in range(N + 1):
            random.shuffle(words_by_weight[wt])
        
        # Process in order of increasing weight
        for wt in range(1, N + 1):
            if time.time() > deadline:
                break
            for w in words_by_weight[wt]:
                if time.time() > deadline:
                    break
                if not forbidden[w]:
                    # Add this word
                    selected.append(w)
                    forbidden[w] = 1
                    for off in all_offsets:
                        idx = w ^ off
                        if not forbidden[idx]:
                            forbidden[idx] = 1
    elif order == 'weight_desc':
        words_by_weight = [[] for _ in range(N + 1)]
        for w in range(1, TOTAL_WORDS):
            wt = w.bit_count()
            words_by_weight[wt].append(w)
        for wt in range(N + 1):
            random.shuffle(words_by_weight[wt])
        
        for wt in range(N, 0, -1):
            if time.time() > deadline:
                break
            for w in words_by_weight[wt]:
                if time.time() > deadline:
                    break
                if not forbidden[w]:
                    selected.append(w)
                    forbidden[w] = 1
                    for off in all_offsets:
                        idx = w ^ off
                        if not forbidden[idx]:
                            forbidden[idx] = 1
    elif order == 'random':
        candidates = list(range(1, TOTAL_WORDS))
        random.shuffle(candidates)
        for w in candidates:
            if time.time() > deadline:
                break
            if not forbidden[w]:
                selected.append(w)
                forbidden[w] = 1
                for off in all_offsets:
                    idx = w ^ off
                    if not forbidden[idx]:
                        forbidden[idx] = 1
    elif order == 'weight_mid':
        # Start from middle weights (around 9) and expand outward
        words_by_weight = [[] for _ in range(N + 1)]
        for w in range(1, TOTAL_WORDS):
            wt = w.bit_count()
            words_by_weight[wt].append(w)
        for wt in range(N + 1):
            random.shuffle(words_by_weight[wt])
        
        # Order: 9, 8, 10, 7, 11, 6, 12, 5, 13, 4, 14, 3, 15, 2, 16, 1, 17, 18
        mid = N // 2
        order_list = [mid]
        for d in range(1, N // 2 + 1):
            if mid - d >= 0:
                order_list.append(mid - d)
            if mid + d <= N:
                order_list.append(mid + d)
        # Add any remaining
        for wt in range(N + 1):
            if wt not in order_list:
                order_list.append(wt)
        
        for wt in order_list:
            if time.time() > deadline:
                break
            for w in words_by_weight[wt]:
                if time.time() > deadline:
                    break
                if not forbidden[w]:
                    selected.append(w)
                    forbidden[w] = 1
                    for off in all_offsets:
                        idx = w ^ off
                        if not forbidden[idx]:
                            forbidden[idx] = 1
    
    return selected

def _local_search(selected, deadline, seed=0):
    """
    Local search: try to find a word that can be added by temporarily removing one selected word.
    Or try swap: remove one, add one or more.
    """
    random.seed(seed)
    start_time = time.time()
    
    all_offsets, offsets_by_weight = _precompute_neighbors()
    
    # Count how many selected words forbid each position
    # This is expensive; let's do a simpler approach
    # Try: for each selected word, see if removing it allows adding 2+ new words
    
    # Simpler: just try to add more with a different order starting from current set
    # Actually, let's try a different approach:
    # Rebuild the forbidden set from scratch and try different orderings
    
    # For now, just return the input (no-op local search)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with Reed-Muller base (32 words)
    base_words = _reed_muller_base()
    best_code = list(base_words)
    best_len = len(best_code)
    
    # Try multiple greedy strategies with bitset approach
    orders = ['weight_asc', 'weight_desc', 'weight_mid', 'random']
    seeds = [42, 123, 7, 99, 256, 1024]
    
    # We have limited time. Each bitset greedy run should be fast.
    # Let's try a few combinations.
    
    for order in orders:
        for seed in seeds[:3]:  # Limit seeds per order to save time
            if time.time() > deadline:
                break
            code = _greedy_bitset(deadline, seed=seed, order=order)
            if len(code) > best_len:
                best_code = code
                best_len = len(code)
    
    # If we still have time, try more seeds with the best-performing order
    if time.time() < deadline - 10:
        # Determine which order worked best
        # For now, just do a few more random-order runs
        for seed in seeds[3:]:
            if time.time() > deadline:
                break
            code = _greedy_bitset(deadline, seed=seed, order='random')
            if len(code) > best_len:
                best_code = code
                best_len = len(code)
    
    # Final: try weight_mid with more seeds if time allows
    if time.time() < deadline - 5:
        for seed in range(100, 110):
            if time.time() > deadline:
                break
            code = _greedy_bitset(deadline, seed=seed, order='weight_mid')
            if len(code) > best_len:
                best_code = code
                best_len = len(code)
    
    return best_code
# EVOLVE-BLOCK-END
