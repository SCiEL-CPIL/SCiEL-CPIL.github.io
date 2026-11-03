<<<<<<< SEARCH
import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # Leave buffer for safety

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    # Map each t-subset to an index for fast counting
    t_subset_to_idx = {frozenset(s): i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    def get_idx(s):
        return t_subset_to_idx[frozenset(s)]

    def _greedy_cover(v, k, t, rng):
        """Fast greedy construction to get a valid starting covering."""
        uncovered = set(range(num_t_subsets))
        blocks = []
        block_set = set()
        
        while uncovered:
            # Pick a random uncovered t-subset as seed
            seed_idx = rng.choice(tuple(uncovered))
            seed = list(all_t_subsets[seed_idx])
            block = list(seed)
            rng.shuffle(block)
            
            # Greedily add points to maximize coverage
            while len(block) < k:
                best_x = None
                best_gain = -1
                cands = [x for x in range(v) if x not in block]
                rng.shuffle(cands)
                
                for x in cands:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        fs = frozenset(list(rest) + [x])
                        idx = t_subset_to_idx.get(fs)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            
            block.sort()
            b_tuple = tuple(block)
            if b_tuple not in block_set:
                blocks.append(b_tuple)
                block_set.add(b_tuple)
                for c in itertools.combinations(block, t):
                    idx = t_subset_to_idx[frozenset(c)]
                    uncovered.discard(idx)
        
        return blocks
=======
import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # Leave buffer for safety

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    # Map each t-subset to an index for fast counting
    t_subset_to_idx = {frozenset(s): i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    def get_idx(s):
        return t_subset_to_idx[frozenset(s)]

    def _scarcity_weighted_greedy(v, k, t, rng):
        """Scarcity-weighted greedy: prioritize blocks that cover rare/hard triples."""
        uncovered = set(range(num_t_subsets))
        blocks = []
        block_set = set()
        
        # Track point frequencies to compute scarcity weights
        point_freq = [0] * v
        
        while uncovered:
            # Compute scarcity weight for each uncovered triple
            # A triple is "hard" if its points have low frequency
            best_seed_idx = None
            best_weight = -1
            sample_size = min(50, len(uncovered))
            sample = rng.sample(list(uncovered), sample_size)
            
            for seed_idx in sample:
                seed = all_t_subsets[seed_idx]
                # Weight = product of (v - freq) for each point (higher = rarer)
                weight = 1.0
                for p in seed:
                    weight *= (v - point_freq[p])
                if weight > best_weight:
                    best_weight = weight
                    best_seed_idx = seed_idx
            
            seed = list(all_t_subsets[best_seed_idx])
            block = list(seed)
            rng.shuffle(block)
            
            # Update point frequencies for the seed
            for p in seed:
                point_freq[p] += 1
            
            # Greedily add points to maximize coverage with scarcity weighting
            while len(block) < k:
                best_x = None
                best_gain = -1
                cands = [x for x in range(v) if x not in block]
                rng.shuffle(cands)
                
                for x in cands:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        fs = frozenset(list(rest) + [x])
                        idx = t_subset_to_idx.get(fs)
                        if idx is not None and idx in uncovered:
                            # Weight by scarcity of the resulting triple
                            triple = all_t_subsets[idx]
                            scarcity = 1.0
                            for p in triple:
                                scarcity *= (v - point_freq[p])
                            gain += scarcity
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
                point_freq[best_x] += 1
            
            block.sort()
            b_tuple = tuple(block)
            if b_tuple not in block_set:
                blocks.append(b_tuple)
                block_set.add(b_tuple)
                for c in itertools.combinations(block, t):
                    idx = t_subset_to_idx[frozenset(c)]
                    uncovered.discard(idx)
        
        return blocks
>>>>>>> REPLACE

