"""Snake-in-the-box Q_9: bit-parallel greedy + beam search."""
import time
import random
import numpy as np

N = 9
SIZE = 1 << N  # 512
TIME_LIMIT = 110  # safety margin below 120s hard kill

# Precompute neighbor masks: for each vertex, the 512-bit mask of its 9 neighbors
NEIGHBOR_MASKS = np.zeros(SIZE, dtype=np.uint64)  # actually need 512 bits -> use Python int
NEIGHBOR_MASKS = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        nb = v ^ (1 << b)
        m |= (1 << nb)
    NEIGHBOR_MASKS[v] = m

# Precompute single-bit masks for each vertex (for the "used" set)
VERTEX_MASKS = [1 << v for v in range(SIZE)]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    def greedy_bitmask(start_vertex, bit_order, max_steps=512):
        """Greedy extension using 512-bit integer bitmask for used+forbidden."""
        path = [start_vertex]
        # bitmask: bit v is set if vertex v is used OR forbidden
        mask = VERTEX_MASKS[start_vertex]
        # Initially, neighbors of start_vertex are forbidden (they'd be non-consecutive to it)
        # Actually for a path of length 1, no non-consecutive constraint yet.
        # But as we extend, the previous vertex's neighbors become forbidden.
        # Let's think: when we add vertex v to the end, the new forbidden set
        # includes all neighbors of the second-to-last vertex (the one before v),
        # because those neighbors would be at distance 1 from a non-consecutive vertex.
        # Wait, more precisely: if path is [..., a, b, v], then any neighbor of a
        # (other than b) cannot be added later because it would differ from a by 1 bit
        # and not be consecutive to a.
        # So when we commit v (with previous being b, and the one before b being a),
        # we add neighbors of a to forbidden.
        # Actually simpler: when we move from b to v, the vertex b becomes "internal"
        # (no longer an endpoint), so all neighbors of b that aren't already in the path
        # become forbidden.
        # Let me re-think:
        # Path: p0, p1, p2, ..., pk
        # p0 and pk are endpoints. Internal vertices p1..p_{k-1} have all their neighbors
        # (except the two path-neighbors) forbidden for future extension.
        # When we extend from pk to pk+1, pk becomes internal, so neighbors of pk
        # (except pk-1 and pk+1) become forbidden.
        # But we also need to forbid pk+1 itself (it's used).
        # And we need to check that pk+1 is not already forbidden.
        
        # Revised approach:
        # mask tracks: used vertices + forbidden vertices
        # When extending from current (last vertex) to new vertex:
        #   - new vertex must not be in mask
        #   - after adding, the old last vertex becomes internal:
        #     all its neighbors (except the one before it in path and the new vertex) become forbidden
        #   - but actually, ALL neighbors of the old last vertex that aren't the previous
        #     path vertex and aren't the new vertex become forbidden
        #   - Wait, the previous path vertex is already used, so it's in mask.
        #   - The new vertex will be in mask.
        #   - So: add all neighbors of old_last to mask.
        
        for step in range(max_steps):
            if time_left() < 2.0:
                break
            cur = path[-1]
            cur_mask = NEIGHBOR_MASKS[cur]
            # Available neighbors: those not in mask
            avail = cur_mask & ~mask
            if avail == 0:
                break
            
            # Score each available neighbor: count its available neighbors (2-step lookahead)
            best_v = -1
            best_score = -1
            # Iterate through set bits in avail
            a = avail
            while a:
                # Get lowest set bit
                lb = a & (-a)
                v = lb.bit_length() - 1  # index of the bit
                a ^= lb
                
                # Score: how many neighbors of v are available (not in mask, not cur)
                v_mask = NEIGHBOR_MASKS[v] & ~mask & ~VERTEX_MASKS[cur]
                score = v_mask.bit_count()
                
                # Tie-break: prefer lower popcount vertex (more "room")
                if score > best_score or (score == best_score and (best_v == -1 or v.bit_count() < best_v.bit_count())):
                    best_score = score
                    best_v = v
            
            if best_v == -1:
                break
            
            # Commit: add best_v to path and mask
            path.append(best_v)
            # Old last vertex (cur) becomes internal: all its neighbors become forbidden
            # (the previous path vertex is already in mask as used)
            mask |= NEIGHBOR_MASKS[cur]
            # New vertex is used
            mask |= VERTEX_MASKS[best_v]
        
        return path

    def greedy_bitmask_randomized(start_vertex, bit_order, max_steps=512, noise=0.3):
        """Greedy with randomized tie-breaking for diversity."""
        path = [start_vertex]
        mask = VERTEX_MASKS[start_vertex]
        
        for step in range(max_steps):
            if time_left() < 2.0:
                break
            cur = path[-1]
            cur_mask = NEIGHBOR_MASKS[cur]
            avail = cur_mask & ~mask
            if avail == 0:
                break
            
            # Collect candidates with scores
            candidates = []
            a = avail
            while a:
                lb = a & (-a)
                v = lb.bit_length() - 1
                a ^= lb
                v_mask = NEIGHBOR_MASKS[v] & ~mask & ~VERTEX_MASKS[cur]
                score = v_mask.bit_count()
                candidates.append((score, v.bit_count(), v))
            
            if not candidates:
                break
            
            # Sort by score desc, then popcount asc
            candidates.sort(key=lambda x: (-x[0], x[1]))
            
            # With some probability, pick from top-k instead of just best
            k = min(len(candidates), max(1, int(noise * len(candidates))))
            if random.random() < noise and len(candidates) > 1:
                best_v = candidates[random.randint(0, k - 1)][2]
            else:
                best_v = candidates[0][2]
            
            path.append(best_v)
            mask |= NEIGHBOR_MASKS[cur]
            mask |= VERTEX_MASKS[best_v]
        
        return path

    def beam_search_bitmask(start_vertex, bit_order, beam_width=40, max_steps=512):
        """Beam search using bitmask representation."""
        # Each beam: (path_list, mask_int)
        beams = [([start_vertex], VERTEX_MASKS[start_vertex])]
        
        for step in range(max_steps):
            if time_left() < 3.0:
                break
            new_beams = []
            for (bp, bm) in beams:
                cur = bp[-1]
                cur_mask = NEIGHBOR_MASKS[cur]
                avail = cur_mask & ~bm
                if avail == 0:
                    continue
                
                # Score candidates
                scored = []
                a = avail
                while a:
                    lb = a & (-a)
                    v = lb.bit_length() - 1
                    a ^= lb
                    v_mask = NEIGHBOR_MASKS[v] & ~bm & ~VERTEX_MASKS[cur]
                    score = v_mask.bit_count()
                    scored.append((-score, v.bit_count(), v))
                
                scored.sort()
                
                # Take top candidates per beam
                take = max(1, beam_width // max(1, len(beams)))
                for (_, _, v) in scored[:take]:
                    new_bp = bp + [v]
                    new_bm = bm | NEIGHBOR_MASKS[cur] | VERTEX_MASKS[v]
                    new_beams.append((new_bp, new_bm))
            
            if not new_beams:
                break
            
            # Keep top beam_width by length (all same length here), break ties by mask popcount (fewer forbidden = more room)
            new_beams.sort(key=lambda x: (len(x[0]), -x[1].bit_count()))
            beams = new_beams[:beam_width]
        
        if not beams:
            return [start_vertex]
        best = max(beams, key=lambda x: len(x[0]))
        return best[0]

    def local_improvement_bitmask(path):
        """Try to extend from both ends using bitmask."""
        if len(path) < 2:
            return path
        
        best_path = list(path)
        
        # Try extending forward
        for _ in range(10):
            cur = best_path[-1]
            # Build mask: all used + all forbidden
            mask = 0
            for v in best_path:
                mask |= VERTEX_MASKS[v]
            # Internal vertices: all except first and last
            for i in range(1, len(best_path) - 1):
                mask |= NEIGHBOR_MASKS[best_path[i]]
            
            avail = NEIGHBOR_MASKS[cur] & ~mask
            if avail == 0:
                break
            # Pick best
            a = avail
            best_v = -1
            best_score = -1
            while a:
                lb = a & (-a)
                v = lb.bit_length() - 1
                a ^= lb
                v_mask = NEIGHBOR_MASKS[v] & ~mask & ~VERTEX_MASKS[cur]
                score = v_mask.bit_count()
                if score > best_score:
                    best_score = score
                    best_v = v
            if best_v == -1:
                break
            best_path.append(best_v)
        
        # Try extending backward
        for _ in range(10):
            cur = best_path[0]
            mask = 0
            for v in best_path:
                mask |= VERTEX_MASKS[v]
            for i in range(1, len(best_path) - 1):
                mask |= NEIGHBOR_MASKS[best_path[i]]
            
            avail = NEIGHBOR_MASKS[cur] & ~mask
            if avail == 0:
                break
            a = avail
            best_v = -1
            best_score = -1
            while a:
                lb = a & (-a)
                v = lb.bit_length() - 1
                a ^= lb
                v_mask = NEIGHBOR_MASKS[v] & ~mask & ~VERTEX_MASKS[cur]
                score = v_mask.bit_count()
                if score > best_score:
                    best_score = score
                    best_v = v
            if best_v == -1:
                break
            best_path.insert(0, best_v)
        
        return best_path

    def verify_snake(path):
        """Verify validity."""
        if not path:
            return False
        used = set(path)
        if len(used) != len(path):
            return False
        for i in range(len(path)):
            for j in range(i + 1, len(path)):
                diff = path[i] ^ path[j]
                if abs(i - j) == 1:
                    if diff == 0 or (diff & (diff - 1)) != 0:
                        return False
                else:
                    if diff == 0 or (diff & (diff - 1)) == 0:
                        return False
        return True

    best_path = None
    best_len = 0

    # Phase 1: Many fast greedy attempts (60% of time)
    deadline1 = start_time + time_left() * 0.6

    # First: deterministic greedy from 0
    p = greedy_bitmask(0, list(range(N)))
    best_path = p
    best_len = len(p)

    attempts = 0
    while time.time() < deadline1:
        sv = random.randint(0, SIZE - 1)
        bo = list(range(N))
        random.shuffle(bo)
        # Alternate between pure greedy and randomized
        if attempts % 3 == 0:
            p = greedy_bitmask(sv, bo)
        elif attempts % 3 == 1:
            p = greedy_bitmask_randomized(sv, bo, noise=0.2)
        else:
            p = greedy_bitmask_randomized(sv, bo, noise=0.5)
        attempts += 1
        if len(p) > best_len:
            best_len = len(p)
            best_path = p

    # Phase 2: Beam search (30% of time)
    deadline2 = start_time + time_left() * 0.7
    beam_widths = [30, 40, 50, 60]
    bw_idx = 0
    while time.time() < deadline2:
        sv = random.randint(0, SIZE - 1)
        bo = list(range(N))
        random.shuffle(bo)
        bw = beam_widths[bw_idx % len(beam_widths)]
        bw_idx += 1
        p = beam_search_bitmask(sv, bo, beam_width=bw)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p

    # Phase 3: Local improvement on best path
    if time_left() > 5.0:
        improved = local_improvement_bitmask(best_path)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)

    # Phase 4: More beam search with larger widths if time permits
    if time_left() > 10.0:
        for bw in [50, 70, 100]:
            if time_left() < 8.0:
                break
            sv = random.randint(0, SIZE - 1)
            bo = list(range(N))
            random.shuffle(bo)
            p = beam_search_bitmask(sv, bo, beam_width=bw)
            if len(p) > best_len:
                best_len = len(p)
                best_path = p

    # Final local improvement
    if time_left() > 3.0:
        improved = local_improvement_bitmask(best_path)
        if len(improved) > best_len:
            best_path = improved

    # Safety: verify before returning
    if best_path and not verify_snake(best_path):
        # Fallback: simple valid path
        best_path = [0]
    
    return best_path
# EVOLVE-BLOCK-END
