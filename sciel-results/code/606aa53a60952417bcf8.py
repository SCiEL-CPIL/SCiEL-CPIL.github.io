"""Snake-in-the-box Q_9: beam-search + greedy + local improvement."""
import time
import random
import numpy as np

N = 9
TIME_LIMIT = 110  # safety margin below 120s hard kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return TIME_LIMIT - elapsed()
    
    def verify_snake(path):
        """Verify that path is a valid induced path in Q_N."""
        if not path:
            return False
        used = set(path)
        if len(used) != len(path):
            return False
        for i in range(len(path)):
            for j in range(i + 1, len(path)):
                diff = path[i] ^ path[j]
                # Consecutive must differ in exactly 1 bit
                # Non-consecutive must differ in at least 2 bits
                if abs(i - j) == 1:
                    if diff == 0 or (diff & (diff - 1)) != 0:
                        return False
                else:
                    # Must not differ in exactly 1 bit
                    if diff == 0 or (diff & (diff - 1)) == 0:
                        return False
        return True
    
    def get_neighbors(v):
        """Generate all N neighbors of v."""
        for b in range(N):
            yield v ^ (1 << b)
    
    def greedy_extend(path, used, forbidden, bit_order, max_steps=None):
        """Extend path greedily using forbidden set. Returns extended path."""
        if max_steps is None:
            max_steps = 512
        for _ in range(max_steps):
            cur = path[-1]
            found = False
            for b in bit_order:
                v = cur ^ (1 << b)
                if v not in used and v not in forbidden:
                    # Accept v, update forbidden
                    for nb in get_neighbors(cur):
                        forbidden.add(nb)
                    path.append(v)
                    used.add(v)
                    found = True
                    break
            if not found:
                break
        return path
    
    def greedy_with_lookahead(path, used, forbidden, bit_order, max_steps=None, lookahead_depth=2):
        """Greedy with lookahead scoring. Returns extended path."""
        if max_steps is None:
            max_steps = 512
        for _ in range(max_steps):
            cur = path[-1]
            best_candidate = None
            best_score = -1
            
            candidates = []
            for b in bit_order:
                v = cur ^ (1 << b)
                if v not in used and v not in forbidden:
                    candidates.append(v)
            
            if not candidates:
                break
            
            if lookahead_depth <= 1 or len(candidates) <= 2:
                best_candidate = candidates[0]
            else:
                for v in candidates:
                    # Score: count of valid next moves from v, weighted by their own options
                    score = 0
                    for b2 in bit_order:
                        v2 = v ^ (1 << b2)
                        if v2 == cur:
                            continue
                        if v2 not in used and v2 not in forbidden:
                            # Count how many options v2 has (3-step lookahead)
                            sub_score = 0
                            for b3 in bit_order:
                                v3 = v2 ^ (1 << b3)
                                if v3 == v:
                                    continue
                                if v3 not in used and v3 not in forbidden:
                                    sub_score += 1
                            score += 1 + 0.5 * sub_score
                    if score > best_score:
                        best_score = score
                        best_candidate = v
            
            if best_candidate is None:
                break
            
            # Commit
            for nb in get_neighbors(cur):
                forbidden.add(nb)
            path.append(best_candidate)
            used.add(best_candidate)
        
        return path
    
    def beam_search_extend(path, used, forbidden, bit_order, beam_width=50, max_steps=512):
        """Beam search with 2-step lookahead scoring for better path selection."""
        beams = [(list(path), set(used), set(forbidden))]
        
        for step in range(max_steps):
            if time_left() < 3.0:
                break
            new_beams = []
            for (bp, bu, bf) in beams:
                cur = bp[-1]
                candidates = []
                for b in bit_order:
                    v = cur ^ (1 << b)
                    if v not in bu and v not in bf:
                        # Score: count valid next moves from v
                        score = 0
                        for b2 in bit_order:
                            v2 = v ^ (1 << b2)
                            if v2 == cur:
                                continue
                            if v2 not in bu and v2 not in bf:
                                score += 1
                        candidates.append((score, v))
                
                if not candidates:
                    continue
                
                # Sort by score descending
                candidates.sort(key=lambda x: -x[0])
                
                # Take top candidates per beam to maintain diversity
                take = max(1, min(len(candidates), beam_width // max(1, len(beams))))
                for (s, v) in candidates[:take]:
                    new_bp = bp + [v]
                    new_bu = set(bu)
                    new_bu.add(v)
                    new_bf = set(bf)
                    for nb in get_neighbors(cur):
                        new_bf.add(nb)
                    new_beams.append((new_bp, new_bu, new_bf))
            
            if not new_beams:
                break
            
            # Keep top beam_width by length, then by last-score heuristic
            new_beams.sort(key=lambda x: -len(x[0]))
            beams = new_beams[:beam_width]
        
        # Return the longest path from beams
        best = max(beams, key=lambda x: len(x[0]))
        return best[0]
    
    def local_improvement(path, bit_order, max_rounds=3):
        """Try to improve path by local modifications."""
        if len(path) <= 2:
            return path
        
        best_path = list(path)
        best_len = len(best_path)
        
        for round_num in range(max_rounds):
            if time_left() < 5.0:
                break
            
            improved = False
            # Try inserting vertices at various positions
            # Try extending from both ends
            for direction in [1, -1]:
                cur = best_path[-1] if direction == 1 else best_path[0]
                used = set(best_path)
                
                # Compute forbidden: neighbors of all non-end vertices
                if direction == 1:
                    # Extending forward: forbidden = neighbors of best_path[0:-1]
                    forbidden = set()
                    for i in range(len(best_path) - 1):
                        for nb in get_neighbors(best_path[i]):
                            forbidden.add(nb)
                else:
                    # Extending backward: forbidden = neighbors of best_path[1:]
                    forbidden = set()
                    for i in range(1, len(best_path)):
                        for nb in get_neighbors(best_path[i]):
                            forbidden.add(nb)
                
                # Try to find a vertex adjacent to cur that's not used/forbidden
                for b in bit_order:
                    v = cur ^ (1 << b)
                    if v not in used and v not in forbidden:
                        if direction == 1:
                            best_path.append(v)
                        else:
                            best_path.insert(0, v)
                        improved = True
                        break
            
            if not improved:
                break
            
            if len(best_path) > best_len:
                best_len = len(best_path)
        
        return best_path
    
    def try_greedy_with_lookahead(start_vertex, bit_order, max_steps=512):
        """Greedy with simple lookahead using forbidden set."""
        path = [start_vertex]
        used = {start_vertex}
        forbidden = set()
        
        for _ in range(max_steps):
            cur = path[-1]
            best_candidate = None
            best_score = -1

            for b in bit_order:
                v = cur ^ (1 << b)
                if v in used or v in forbidden:
                    continue
                
                # Score: count valid extensions from v
                score = 0
                for b2 in bit_order:
                    v2 = v ^ (1 << b2)
                    if v2 == cur:
                        continue
                    if v2 not in used and v2 not in forbidden:
                        score += 1
                        
                if score > best_score:
                    best_score = score
                    best_candidate = v

            if best_candidate is None:
                break

            # Commit to best_candidate
            for nb in get_neighbors(cur):
                forbidden.add(nb)
                
            path.append(best_candidate)
            used.add(best_candidate)

        return path
    
    def random_bit_order():
        order = list(range(N))
        random.shuffle(order)
        return order
    
    def run_greedy_attempt(sv, bo, use_lookahead=True, max_steps=512):
        path = [sv]
        used = {sv}
        forbidden = set()
        if use_lookahead:
            return greedy_with_lookahead(path, used, forbidden, bo, max_steps, lookahead_depth=2)
        else:
            return greedy_extend(path, used, forbidden, bo, max_steps)
    
    def run_beam_attempt(sv, bo, beam_width=30, max_steps=512):
        path = [sv]
        used = {sv}
        forbidden = set()
        return beam_search_extend(path, used, forbidden, bo, beam_width, max_steps)
    
    best_path = None
    best_len = 0
    
    # Phase 1: Many fast greedy attempts with random restarts
    phase1_time = min(25.0, time_left() * 0.35)
    deadline1 = start_time + phase1_time
    
    # First: try deterministic baseline
    p = run_greedy_attempt(0, list(range(N)), use_lookahead=True)
    best_path = p
    best_len = len(p)
    
    # Random restarts with greedy + lookahead
    attempts = 0
    while time.time() < deadline1:
        sv = random.randint(0, 2**N - 1)
        bo = random_bit_order()
        p = run_greedy_attempt(sv, bo, use_lookahead=True)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
        attempts += 1
    
    # Phase 2: Beam search attempts with remaining time
    phase2_deadline = start_time + time_left() * 0.55
    
    beam_widths = [20, 30, 40, 50]
    bw_idx = 0
    
    while time.time() < phase2_deadline:
        sv = random.randint(0, 2**N - 1)
        bo = random_bit_order()
        bw = beam_widths[bw_idx % len(beam_widths)]
        bw_idx += 1
        p = run_beam_attempt(sv, bo, beam_width=bw, max_steps=512)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Phase 3: Local improvement on best path
    if time_left() > 5.0:
        bo = random_bit_order()
        improved = local_improvement(best_path, bo, max_rounds=5)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)
    
    # Phase 4: More greedy attempts with the best bit order
    if time_left() > 3.0:
        bo_best = random_bit_order()
        for _ in range(20):
            if time_left() < 2.0:
                break
            sv = random.randint(0, 2**N - 1)
            p = run_greedy_attempt(sv, bo_best, use_lookahead=True)
            if len(p) > best_len:
                best_len = len(p)
                best_path = p
    
    # Final: try a few more beam searches with larger width if time allows
    if time_left() > 8.0:
        for bw in [40, 50, 60]:
            if time_left() < 6.0:
                break
            sv = random.randint(0, 2**N - 1)
            bo = random_bit_order()
            p = run_beam_attempt(sv, bo, beam_width=bw, max_steps=512)
            if len(p) > best_len:
                best_len = len(p)
                best_path = p
    
    # Return the best path found
    return best_path
# EVOLVE-BLOCK-END
