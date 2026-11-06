"""
Rewritten solver for the dense increasing sequence with unique consecutive block products.

Strategy:
1. Greedy construction (scan 2..128 in order, include x if all block products remain unique).
2. Local improvement: try removing one element and re-filling greedily to see if a longer sequence is found.
3. Deadline guard to return best result within 60s.
"""

import time
import sys

def _all_block_products(seq):
    """Return a set of all nonempty consecutive block products for the given sorted sequence."""
    n = len(seq)
    prods = set()
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            prods.add(p)
    return prods

def _is_valid(seq):
    """Check if all consecutive block products are distinct."""
    n = len(seq)
    prods = []
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            prods.append(p)
    return len(prods) == len(set(prods))

def _greedy_build(start_seq, deadline):
    """
    Starting from a sorted list start_seq, greedily try to add each number from 2 to 128
    (inserting in sorted order) if the resulting sequence has all distinct block products.
    Returns the resulting sequence.
    """
    seq = list(start_seq)
    seq_set = set(seq)
    
    for x in range(2, 129):
        if time.time() > deadline:
            break
        if x in seq_set:
            continue
        # Find insertion position
        # Binary search for position
        lo, hi = 0, len(seq)
        while lo < hi:
            mid = (lo + hi) // 2
            if seq[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        pos = lo
        new_seq = seq[:pos] + [x] + seq[pos:]
        if _is_valid(new_seq):
            seq = new_seq
            seq_set.add(x)
    return seq

def _local_improve(seq, deadline):
    """
    Try removing each element from seq and re-filling greedily.
    Return the best sequence found (longest, then lexicographically smallest).
    """
    best = list(seq)
    n = len(seq)
    for i in range(n):
        if time.time() > deadline:
            break
        # Remove element at position i
        reduced = seq[:i] + seq[i+1:]
        # Greedily fill from reduced
        candidate = _greedy_build(reduced, deadline)
        if len(candidate) > len(best):
            best = candidate
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s
    
    # Phase 1: Initial greedy construction
    seq = _greedy_build([], deadline)
    
    # Phase 2: Local improvement - try removing one element and re-filling
    improved = _local_improve(seq, deadline)
    if len(improved) > len(seq):
        seq = improved
    
    # Phase 3: Another round of local improvement if time permits
    if time.time() < deadline - 10:
        improved2 = _local_improve(seq, deadline)
        if len(improved2) > len(seq):
            seq = improved2
    
    return seq
