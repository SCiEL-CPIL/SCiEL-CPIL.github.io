"""Stronger covering design algorithm for C(v, k, t) with deadline guard."""
import itertools
import time
import random
import heapq
from collections import defaultdict

def construct(v, k, t):
    # Deadline guard: must return well before 150s wall-clock
    deadline = time.time() + 140  # 10s safety margin
    
    # Total number of t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    total_t_subsets = len(all_t_subsets)
    uncovered = set(all_t_subsets)
    blocks = []
    
    # Precompute: for each k-subset, which t-subsets it covers
    # For C(19,8,5): C(19,8)=75582 k-subsets, each covers C(8,5)=56 t-subsets
    # Total incidence: 75582 * 56 = 4,232,592 entries - manageable
    
    # Strategy: Global greedy set cover
    # At each step, find the k-subset that covers the most uncovered t-subsets
    # Use a priority queue for efficiency
    
    # Precompute all k-subsets and their covered t-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    
    # For each k-subset, precompute which t-subsets it covers
    k_to_t = {}
    for ks in all_k_subsets:
        k_to_t[ks] = tuple(itertools.combinations(ks, t))
    
    # Build reverse index: for each t-subset, which k-subsets cover it
    t_to_k = defaultdict(list)
    for ks, ts_list in k_to_t.items():
        for ts in ts_list:
            t_to_k[ts].append(ks)
    
    # Count how many k-subsets cover each t-subset (for bottleneck heuristic)
    t_degree = {ts: len(ks_list) for ts, ks_list in t_to_k.items()}
    
    # Priority-based greedy: repeatedly pick the k-subset with max coverage of uncovered
    # Use a heap where we store (-gain, ks)
    
    # To speed up, we'll use a "lazy" approach:
    # 1. Start with all k-subsets in a heap with their initial coverage
    # 2. When we pick a k-subset, remove its covered t-subsets from uncovered
    # 3. Recompute gains for remaining k-subsets lazily
    
    # Initial: compute gain for each k-subset
    # gain(ks) = |{ts in k_to_t[ks] : ts in uncovered}|
    
    # For efficiency with C(19,8,5), let's use a different approach:
    # Group k-subsets by how many uncovered t-subsets they cover
    # Use a bucket-based approach
    
    # Actually, let's use a simpler but effective approach:
    # Randomized greedy with multiple restarts and local improvement
    
    best_blocks = None
    best_count = float('inf')
    
    start_time = time.time()
    max_time = 130  # Leave 10s for final processing
    
    # Phase 1: Multiple greedy constructions with different random seeds
    n_restarts = 5
    rng = random.Random(42)
    
    for restart in range(n_restarts):
        if time.time() - start_time > max_time - 20:
            break
        
        uncovered = set(all_t_subsets)
        blocks = []
        
        # Greedy: repeatedly pick the k-subset that covers the most uncovered t-subsets
        # To make this faster, we'll use a heuristic:
        # 1. Pick the hardest uncovered t-subset (fewest covering k-subsets)
        # 2. Among k-subsets covering it, pick the one with max total uncovered coverage
        
        while uncovered and time.time() - start_time < max_time - 15:
            # Find the uncovered t-subset with minimum degree (hardest to cover)
            # For efficiency, sample a few candidates
            sample_size = min(50, len(uncovered))
            candidates = rng.sample(list(uncovered), sample_size) if len(uncovered) > sample_size else list(uncovered)
            
            best_ts = min(candidates, key=lambda ts: t_degree[ts])
            
            # Among all k-subsets covering best_ts, pick the one with max uncovered coverage
            best_ks = None
            best_gain = -1
            best_covered = set()
            
            for ks in t_to_k[best_ts]:
                # Count how many uncovered t-subsets this k-subset covers
                gain = 0
                covered = set()
                for ts in k_to_t[ks]:
                    if ts in uncovered:
                        gain += 1
                        covered.add(ts)
                
                if gain > best_gain:
                    best_gain = gain
                    best_ks = ks
                    best_covered = covered
            
            if best_ks is not None and best_gain > 0:
                blocks.append(list(best_ks))
                uncovered -= best_covered
            else:
                # No k-subset covers this t-subset well, try another approach
                # Just pick a random k-subset that covers at least one uncovered t-subset
                found = False
                for _ in range(100):
                    ks = rng.choice(all_k_subsets)
                    gain = 0
                    covered = set()
                    for ts in k_to_t[ks]:
                        if ts in uncovered:
                            gain += 1
                            covered.add(ts)
                    if gain > 0:
                        blocks.append(list(ks))
                        uncovered -= covered
                        found = True
                        break
                if not found:
                    break
        
        # Phase 2: Pruning - remove redundant blocks
        if time.time() - start_time < max_time - 5 and len(blocks) > 1:
            # Build coverage count for each t-subset
            t_count = defaultdict(int)
            for block in blocks:
                for ts in itertools.combinations(block, t):
                    t_count[ts] += 1
            
            # Try removing blocks that are redundant
            changed = True
            while changed and time.time() - start_time < max_time - 3:
                changed = False
                for i in range(len(blocks) - 1, -1, -1):
                    if time.time() - start_time > max_time - 3:
                        break
                    # Check if removing block i leaves all t-subsets covered
                    can_remove = True
                    for ts in itertools.combinations(blocks[i], t):
                        if t_count[ts] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        for ts in itertools.combinations(blocks[i], t):
                            t_count[ts] -= 1
                        blocks.pop(i)
                        changed = True
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
        
        # Early stop if we hit the theoretical lower bound
        if best_count <= 264:
            break
    
    return best_blocks if best_blocks else []
