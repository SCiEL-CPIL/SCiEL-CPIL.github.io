import time
import random
from math import gcd
from itertools import combinations

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    a, b = dy, -dx
    d = gcd(abs(a), abs(b))
    if d > 0:
        a //= d
        b //= d
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = -(a * x1 + b * y1)
    return (a, b, c)

def fast_count(points):
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
    
    four_count = 0
    valid = True
    for count in line_counts.values():
        if count >= 5:
            valid = False
            break
        if count == 4:
            four_count += 1
    
    if not valid:
        return -1
    return four_count

def repair_configuration(points, deadline):
    """
    Attempt to fix invalid configurations (5+ collinear) by moving points.
    Returns (points, score).
    """
    current_points = [p[:] for p in points]
    current_score = fast_count(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # If already valid, try to improve
    # If invalid, try to make valid
    
    max_iter = 5000
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # If valid, do standard local search
        if current_score > 0:
             idx = random.randint(0, 15)
             dx = random.randint(-5, 5)
             dy = random.randint(-5, 5)
             new_x = current_points[idx][0] + dx
             new_y = current_points[idx][1] + dy
             
             if abs(new_x) > 1000 or abs(new_y) > 1000:
                 continue
             
             new_pos = [new_x, new_y]
             collision = False
             for i in range(16):
                 if i != idx and current_points[i] == new_pos:
                     collision = True
                     break
             if collision:
                 continue
             
             old_pos = current_points[idx][:]
             current_points[idx] = new_pos
             new_score = fast_count(current_points)
             current_points[idx] = old_pos
             
             if new_score < 0:
                 continue
                 
             if new_score > current_score:
                 current_points[idx] = new_pos
                 current_score = new_score
                 if current_score > best_score:
                     best_score = current_score
                     best_points = [p[:] for p in current_points]
             elif new_score == current_score and random.random() < 0.3:
                 current_points[idx] = new_pos
        else:
            # Invalid: find a line with 5+ points and move one of them
            # Identify the offending line
            n = len(current_points)
            line_counts = {}
            line_points = {}
            for i in range(n):
                for j in range(i + 1, n):
                    key = get_line_key(current_points[i], current_points[j])
                    if key is None:
                        continue
                    if key not in line_counts:
                        line_counts[key] = 0
                        line_points[key] = []
                    line_counts[key] += 1
                    if j not in line_points[key]:
                        line_points[key].append(i)
                    if i not in line_points[key]:
                        line_points[key].append(i)
            
            offending_keys = [k for k, v in line_counts.items() if v >= 5]
            if not offending_keys:
                # Should not happen if fast_count returned -1, but just in case
                continue
                
            key = random.choice(offending_keys)
            pts_on_line = line_points[key]
            # Move one of the points on this line
            idx = random.choice(pts_on_line)
            
            # Try moving it far away
            for _ in range(20):
                dx = random.randint(-50, 50)
                dy = random.randint(-50, 50)
                new_x = current_points[idx][0] + dx
                new_y = current_points[idx][1] + dy
                
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                
                new_pos = [new_x, new_y]
                collision = False
                for i in range(16):
                    if i != idx and current_points[i] == new_pos:
                        collision = True
                        break
                if collision:
                    continue
                
                old_pos = current_points[idx][:]
                current_points[idx] = new_pos
                new_score = fast_count(current_points)
                current_points[idx] = old_pos
                
                if new_score >= 0:
                    current_points[idx] = new_pos
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                    break # Found a valid move, accept it
    
    return best_points, best_score

def line_first_construct(deadline):
    """
    Strategy: Pick random 4-point lines and add their points.
    This directly builds the hypergraph structure.
    """
    best_points = None
    best_score = -1
    
    # Pre-generate a pool of candidate lines
    # We want lines that can hold 4 integer points within [-1000, 1000]
    # Slopes with small numerator/denominator are good for integer points
    candidate_lines = []
    
    # Generate lines with small slopes
    # Line: y = m*x + c or x = c
    # For integer points, if m = p/q, then for x = q*k, y = p*k + c
    # We need 4 such points.
    
    for m_num in range(-20, 21):
        for m_den in range(1, 21):
            if gcd(abs(m_num), m_den) != 1:
                continue
            # Slope is m_num / m_den
            # Points: (m_den * k, m_num * k + c) for k = 0, 1, 2, 3
            # We need to choose c such that points are within bounds
            # And distinct.
            
            # Try different intercepts c
            # c can be any integer.
            # Let's try c in a range that keeps points small
            for c in range(-50, 51):
                pts = []
                valid = True
                for k in range(4):
                    x = m_den * k
                    y = m_num * k + c
                    if abs(x) > 1000 or abs(y) > 1000:
                        valid = False
                        break
                    pts.append((x, y))
                if valid:
                    # Check distinctness (should be if slope != 0 or intercept diff)
                    if len(set(pts)) == 4:
                        candidate_lines.append((m_num, m_den, c, pts))
            
            # Also try vertical lines x = c
            if m_den == 1 and m_num == 0: # Horizontal, already covered?
                pass
            
    # Add vertical lines
    for c in range(-50, 51):
        pts = [(c, y) for y in range(4)]
        candidate_lines.append((0, 0, c, pts)) # Key (0,0) means vertical? No, get_line_key handles it.
        
    # Shuffle
    random.shuffle(candidate_lines)
    
    max_attempts = 100
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        used_points = set()
        selected_lines = []
        
        # Greedy selection of lines
        # Sort candidates by some heuristic? Or just iterate shuffled.
        # We want to pack as many 4-point lines as possible into 16 points.
        # Max number of 4-point lines for 16 points?
        # If we pick a line, we use 4 points.
        # If we pick another line, it can share 0, 1, 2, 3 points.
        # If it shares 2 points, it adds 2 new points.
        # If it shares 3 points, it adds 1 new point (but that makes 5 collinear on the new line? No, the new line has the 3 shared + 1 new = 4. The old line has 4. The intersection is 2 points. Wait.
        # If line L1 has points {A,B,C,D}.
        # Line L2 has points {A,B,E,F}.
        # Points: A,B,C,D,E,F (6 points).
        # L1 is a 4-point line. L2 is a 4-point line.
        # No 5 collinear yet.
        
        # Let's try to build a set of lines.
        
        for line_data in candidate_lines:
            m_num, m_den, c, pts = line_data
            if len(used_points) >= 16:
                break
                
            # Check if this line is "valid" to add
            # 1. All points on this line must not create a 5-collinear on ANY existing line.
            # 2. The points on this line must not create a 5-collinear on THIS line (impossible if we only pick 4).
            # 3. The new points must not collide with existing points unless they are the shared ones.
            
            # Actually, simpler:
            # Add the line's points to the set.
            # Check if the resulting set has any 5-collinear.
            # If yes, reject.
            # If no, accept.
            
            new_points = list(used_points)
            for p in pts:
                if p not in new_points:
                    new_points.append(p)
                    
            if len(new_points) > 16:
                continue
                
            # Check validity
            # We only need to check lines that contain at least 2 points from the new set?
            # Actually, fast_count is O(N^2). N=16 is small.
            # But we are doing this inside a loop of attempts.
            # Let's do a quick check:
            # For each existing line (key) in our "conceptual" set, does it now have 5 points?
            # We don't track all lines, just the ones we selected?
            # No, accidental 5-collinear can happen.
            
            # Let's just run fast_count on the new set of points (capped at 16).
            # This is expensive if done for every candidate line.
            # Optimization: Only check if the new points intersect existing lines.
            
            # Let's just try it. 100 attempts * 1000 lines * fast_count(16) might be too slow?
            # fast_count(16) is ~120 pairs. Very fast.
            # 100 * 1000 * 120 = 12,000,000 operations. Should be fine in Python? Maybe borderline.
            # Let's limit candidate lines checked per attempt.
            
            # Actually, let's just pick a random subset of candidates.
            pass
            
        # This loop structure is a bit messy. Let's restart the logic.
        
        # Better approach for Line-First:
        # 1. Pick a random starting 4-point line.
        # 2. Repeatedly pick a new 4-point line that shares 1 or 2 points with existing points,
        #    such that no 5-collinear is formed.
        # 3. Stop when 16 points are reached or no more lines can be added.
        
        if len(candidate_lines) == 0:
            continue
            
        # Start with a random line
        start_idx = random.randint(0, len(candidate_lines) - 1)
        m_num, m_den, c, pts = candidate_lines[start_idx]
        current_points = set(pts)
        selected_line_keys = [get_line_key(pts[0], pts[1])]
        
        # Try to add more lines
        # We have 4 points. We want to add more.
        # We can add a line that shares 1 point (adds 3), or 2 points (adds 2), or 3 points (adds 1).
        # Sharing 3 points means the new line passes through 3 existing points.
        # If 3 existing points are collinear, the line is determined.
        # If they are not collinear, no line passes through all 3.
        
        # So, we can only add a line sharing 3 points if those 3 points are collinear.
        # In a generic set, 3 points are rarely collinear unless we built it that way.
        # Our starting line has 4 collinear points.
        # So any other line can share at most 2 points with the first line (otherwise it's the same line).
        
        # So, we are looking for lines that share exactly 1 or 2 points with the current set.
        
        # Iterate through candidates
        # To speed up, we can pre-index candidates by their points?
        # Or just iterate.
        
        # Let's iterate through a random permutation of candidates
        cand_indices = list(range(len(candidate_lines)))
        random.shuffle(cand_indices)
        
        for idx in cand_indices:
            if len(current_points) >= 16:
                break
                
            m_num, m_den, c, pts = candidate_lines[idx]
            line_set = set(pts)
            
            # Check overlap
            overlap = line_set.intersection(current_points)
            overlap_size = len(overlap)
            
            # If overlap is 3 or 4, it's the same line or invalid (4 is same line, 3 implies collinear which might be same line or different?)
            # If overlap is 4, same line. Skip.
            # If overlap is 3, the 3 points are collinear (on the new line).
            # Are they collinear on an existing line?
            # If yes, and that existing line is the one we're considering? No.
            # If the 3 points are collinear, they define a line.
            # If that line is the new line, then the new line contains 3 existing points.
            # This is allowed IF the existing line containing them is NOT the same as the new line?
            # Wait, if 3 points are collinear, they lie on a unique line.
            # If that line is the new line, then the new line is the same as the line containing those 3 points.
            # If those 3 points were part of a previous line L_old, then L_old = L_new.
            # So if overlap is 3, it's likely the same line.
            # Let's just check if the line key is already in selected_line_keys.
            
            key = get_line_key(pts[0], pts[1])
            if key in selected_line_keys:
                continue
                
            # If overlap is 0, we add 4 new points.
            # If overlap is 1, we add 3 new points.
            # If overlap is 2, we add 2 new points.
            
            # If overlap >= 3, we are likely duplicating a line or creating a 5-collinear on the old line?
            # If overlap is 3, the new line passes through 3 existing points.
            # If those 3 points are collinear, they are on a line L.
            # If L is not the new line, then the new line intersects L at 3 points -> L = New Line.
            # So overlap 3 implies same line.
            
            if overlap_size > 2:
                continue
                
            # Check if adding these points creates 5-collinear
            # New points to add:
            new_pts = list(line_set - current_points)
            proposed_points = current_points.union(line_set)
            
            # Check validity
            # We can just run fast_count on the proposed points (if size <= 16)
            # But we only add if size <= 16.
            if len(proposed_points) > 16:
                continue
                
            # Check 5-collinear
            # We only need to check lines that contain at least 2 points from the proposed set?
            # Actually, fast_count is robust.
            # Let's just run it.
            score = fast_count(list(proposed_points))
            
            if score >= 0:
                # Valid!
                current_points = proposed_points
                selected_line_keys.append(key)
                
        # End candidate loop
        if len(current_points) == 16:
            pts_list = [list(p) for p in current_points]
            final_score = fast_count(pts_list)
            if final_score > best_score:
                best_score = final_score
                best_points = pts_list
                
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_points = None
    best_score = -1
    
    # Strategy 1: Line-First Construction
    # This is the new structural approach
    lf_points, lf_score = line_first_construct(deadline)
    if lf_points:
        best_points = lf_points
        best_score = lf_score
        
    # Strategy 2: Grid (Fallback / Comparison)
    grid_points = [[x, y] for x in range(4) for y in range(4)]
    grid_score = fast_count(grid_points)
    if grid_score > best_score:
        best_points = grid_points
        best_score = grid_score
        
    # Strategy 3: Local Search from Best
    if best_points:
        # Repair/Improve
        refined_points, refined_score = repair_configuration(best_points, deadline)
        if refined_score > best_score:
            best_points = refined_points
            best_score = refined_score
            
    # If we still have time, try a few more line-first attempts
    while time.time() < deadline - 10.0:
        lf_points, lf_score = line_first_construct(deadline)
        if lf_points and lf_score > best_score:
            best_points = lf_points
            best_score = lf_score
            # Refine
            refined_points, refined_score = repair_configuration(best_points, deadline)
            if refined_score > best_score:
                best_points = refined_points
                best_score = refined_score
                
    # Ensure validity
    if best_points is None:
        return grid_points
        
    final_score = fast_count(best_points)
    if final_score < 0:
        # Fallback to grid if something went wrong
        return grid_points
        
    return best_points

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_points = None
    best_score = -1
    
    # Strategy 1: Line-First Construction
    lf_points, lf_score = line_first_construct(deadline)
    if lf_points:
        best_points = lf_points
        best_score = lf_score
        
    # Strategy 2: Grid (Fallback / Comparison)
    grid_points = [[x, y] for x in range(4) for y in range(4)]
    grid_score = fast_count(grid_points)
    if grid_score > best_score:
        best_points = grid_points
        best_score = grid_score
        
    # Strategy 3: Local Search from Best
    if best_points:
        refined_points, refined_score = repair_configuration(best_points, deadline)
        if refined_score > best_score:
            best_points = refined_points
            best_score = refined_score
            
    # If we still have time, try a few more line-first attempts
    while time.time() < deadline - 10.0:
        lf_points, lf_score = line_first_construct(deadline)
        if lf_points and lf_score > best_score:
            best_points = lf_points
            best_score = lf_score
            refined_points, refined_score = repair_configuration(best_points, deadline)
            if refined_score > best_score:
                best_points = refined_points
                best_score = refined_score
                
    # Ensure validity
    if best_points is None:
        return grid_points
        
    final_score = fast_count(best_points)
    if final_score < 0:
        return grid_points
        
    return best_points
# EVOLVE-BLOCK-END
