"""Improved program for ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys


def _compute_ball(q, n, radius, center):
    """Compute ball as a frozenset of indices for fast membership testing."""
    if radius == 1:
        # Center index
        center_idx = 0
        for i in range(n):
            center_idx = center_idx * q + center[i]
        result = {center_idx}
        # Add neighbors
        for pos in range(n):
            for delta in range(1, q):
                new_val = (center[pos] + delta) % q
                idx = 0
                for j in range(n):
                    val = new_val if j == pos else center[j]
                    idx = idx * q + val
                result.add(idx)
        return frozenset(result)
    else:
        # General radius: enumerate all words within Hamming distance
        result = set()
        # Use recursive generation
        def gen(pos, current, dist):
            if dist > radius:
                return
            if pos == n:
                idx = 0
                for v in current:
                    idx = idx * q + v
                result.add(idx)
                return
            # Don't change this position
            current.append(center[pos])
            gen(pos + 1, current, dist)
            current.pop()
            # Change this position
            for delta in range(1, q):
                current.append((center[pos] + delta) % q)
                gen(pos + 1, current, dist + 1)
                current.pop()
        gen(0, [], 0)
        return frozenset(result)


def _hamming_dist(c1, c2, n):
    """Compute Hamming distance between two codewords."""
    return sum(1 for i in range(n) if c1[i] != c2[i])


def _is_redundant(i, codewords, balls):
    """Check if codeword i is redundant (its ball is covered by others)."""
    ball_i = balls[i]
    for w in ball_i:
        covered_by_other = False
        for j in range(len(codewords)):
            if j == i:
                continue
            if w in balls[j]:
                covered_by_other = True
                break
        if not covered_by_other:
            return False
    return True


def _greedy_cover(q, n, radius, deadline):
    """Greedy set cover: start with all words uncovered, pick codeword covering most uncovered."""
    total_words = q ** n
    uncovered = set(range(total_words))
    codewords = []
    balls = []
    
    # Precompute all possible centers and their balls
    all_centers = list(itertools.product(range(q), repeat=n))
    
    # Sort centers by some heuristic (e.g., lexicographic)
    while uncovered and time.time() < deadline:
        best_center = None
        best_ball = None
        best_count = 0
        
        # Sample a subset of centers for speed
        sample_size = min(len(all_centers), 500)
        sampled = random.sample(all_centers, sample_size)
        
        for center in sampled:
            if time.time() > deadline:
                break
            ball = _compute_ball(q, n, radius, center)
            count = len(ball & uncovered)
            if count > best_count:
                best_count = count
                best_center = center
                best_ball = ball
        
        if best_center is None or best_count == 0:
            # Fallback: try all centers
            for center in all_centers:
                if time.time() > deadline:
                    break
                ball = _compute_ball(q, n, radius, center)
                count = len(ball & uncovered)
                if count > best_count:
                    best_count = count
                    best_center = center
                    best_ball = ball
        
        if best_center is not None and best_count > 0:
            codewords.append(list(best_center))
            balls.append(best_ball)
            uncovered -= best_ball
        else:
            break
    
    return codewords, balls


def _local_search(codewords, balls, q, n, radius, deadline):
    """Local search: try to replace codewords to reduce total count."""
    total_words = q ** n
    
    # Phase 1: Remove redundant codewords
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            if _is_redundant(i, codewords, balls):
                del codewords[i]
                del balls[i]
                changed = True
                break
    
    # Phase 2: Try to replace pairs of codewords with one better codeword
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Compute coverage counts
        coverage = [0] * total_words
        for ball in balls:
            for w in ball:
                coverage[w] += 1
        
        # Find words with coverage exactly 1 (only covered by one codeword)
        # Try to find a codeword that can cover multiple such words
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            
            # Try to find a replacement for codeword i
            # that covers all words uniquely covered by i, plus some others
            unique_to_i = set()
            for w in balls[i]:
                if coverage[w] == 1:
                    unique_to_i.add(w)
            
            if len(unique_to_i) == 0:
                # Can just remove i
                del codewords[i]
                del balls[i]
                improved = True
                break
            
            # Try random replacements
            best_new_cw = None
            best_new_ball = None
            best_gain = 0
            
            for _ in range(200):
                if time.time() > deadline:
                    break
                new_cw = [random.randint(0, q - 1) for _ in range(n)]
                new_ball = _compute_ball(q, n, radius, new_cw)
                
                # Must cover all unique_to_i
                if not unique_to_i.issubset(new_ball):
                    continue
                
                # Count additional coverage
                gain = 0
                for w in new_ball:
                    if w not in balls[i] and coverage[w] == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_new_cw = new_cw
                    best_new_ball = new_ball
            
            if best_new_cw is not None:
                # Replace codeword i with best_new_cw
                codewords[i] = best_new_cw
                balls[i] = best_new_ball
                improved = True
                break
    
    # Phase 3: Final redundancy removal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            if _is_redundant(i, codewords, balls):
                del codewords[i]
                del balls[i]
                changed = True
                break
    
    return codewords, balls


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Strategy: Start with a structured code, then refine
    # For q=3, n=6, radius=1: use a known good starting point
    
    if q == 3 and n == 6 and radius == 1:
        # Use a known construction: take a subset of the 729 words
        # Start with a systematic code and greedily add/remove
        all_words = list(itertools.product(range(3), repeat=6))
        
        # Greedy construction: start empty, add words that cover most uncovered
        uncovered = set(range(729))
        codewords = []
        balls = []
        
        # Precompute all balls
        all_balls = {}
        for w in all_words:
            idx = 0
            for v in w:
                idx = idx * 3 + v
            all_balls[idx] = _compute_ball(3, 6, 1, w)
        
        # Greedy: pick word covering most uncovered
        remaining = list(range(729))
        random.shuffle(remaining)
        
        while uncovered and time.time() < deadline:
            best_idx = -1
            best_count = 0
            best_ball = None
            
            # Check a sample of remaining words
            sample = remaining[:min(len(remaining), 200)]
            for idx in sample:
                if time.time() > deadline:
                    break
                ball = all_balls[idx]
                count = len(ball & uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = idx
                    best_ball = ball
            
            if best_idx == -1 or best_count == 0:
                # Check all remaining
                for idx in remaining:
                    if time.time() > deadline:
                        break
                    ball = all_balls[idx]
                    count = len(ball & uncovered)
                    if count > best_count:
                        best_count = count
                        best_idx = idx
                        best_ball = ball
            
            if best_idx == -1 or best_count == 0:
                break
            
            # Add this codeword
            w = all_words[best_idx]
            codewords.append(list(w))
            balls.append(best_ball)
            uncovered -= best_ball
            remaining.remove(best_idx)
        
        # Local search to improve
        codewords, balls = _local_search(codewords, balls, 3, 6, 1, deadline)
        
        # Final cleanup
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                if _is_redundant(i, codewords, balls):
                    del codewords[i]
                    del balls[i]
                    changed = True
                    break
        
        return codewords
    else:
        # General case: use greedy cover
        codewords, balls = _greedy_cover(q, n, radius, deadline)
        codewords, balls = _local_search(codewords, balls, q, n, radius, deadline)
        return codewords
# EVOLVE-BLOCK-END
