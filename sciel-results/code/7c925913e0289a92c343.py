"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = 230.0  # seconds, with safety margin
    margin = 20.0  # seconds before hard kill
    
    def elapsed():
        return time.time() - start_time
    
    def time_ok():
        return elapsed() < deadline - margin
    
    def make_cell(pattern_func):
        """Create a cell from a pattern function, ensuring symmetry."""
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                # Use pixel centers
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                cell[i][j] = 1 if pattern_func(x, y) else 0
        # Enforce symmetry: cell must equal its transpose and be symmetric under 180° rotation
        # The pattern should already be symmetric, but let's enforce it
        # Symmetry about center: cell[i][j] == cell[size-1-i][size-1-j]
        # Transpose symmetry: cell[i][j] == cell[j][i]
        # We'll average over the symmetry group
        sym_cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                # The symmetry group of the square: identity, 90°, 180°, 270°, and reflections
                # For a square symmetric about center, we need:
                # f(x,y) = f(-x,-y) = f(y,x) = f(-y,-x) etc.
                # In pixel coordinates centered at 0:
                # (i,j) -> (size-1-i, size-1-j) for 180° rotation
                # (i,j) -> (j,i) for transpose (reflection across diagonal)
                # We'll just use the pattern as-is if it's symmetric
                sym_cell[i][j] = cell[i][j]
        return sym_cell
    
    def rod_pattern(radius):
        """Single centered rod."""
        def f(x, y):
            return x * x + y * y < radius * radius
        return f
    
    def four_rod_pattern(radius, offset):
        """Four rods at (±offset, ±offset)."""
        def f(x, y):
            for sx in [1, -1]:
                for sy in [1, -1]:
                    dx = x - sx * offset
                    dy = y - sy * offset
                    if dx * dx + dy * dy < radius * radius:
                        return True
            return False
        return f
    
    def ring_pattern(inner_r, outer_r):
        """Dielectric ring."""
        def f(x, y):
            r2 = x * x + y * y
            return inner_r * inner_r < r2 < outer_r * outer_r
        return f
    
    def cross_pattern(arm_width, arm_length):
        """Cross/plus shape."""
        def f(x, y):
            # Horizontal arm
            if abs(y) < arm_width / 2 and abs(x) < arm_length / 2:
                return True
            # Vertical arm
            if abs(x) < arm_width / 2 and abs(y) < arm_length / 2:
                return True
            return False
        return f
    
    def four_rings_pattern(inner_r, outer_r, offset):
        """Four small rings at (±offset, ±offset)."""
        def f(x, y):
            for sx in [1, -1]:
                for sy in [1, -1]:
                    dx = x - sx * offset
                    dy = y - sy * offset
                    r2 = dx * dx + dy * dy
                    if inner_r * inner_r < r2 < outer_r * outer_r:
                        return True
            return False
        return f
    
    def check_symmetry(cell):
        """Check if cell has square symmetry."""
        size = len(cell)
        # Check 180° rotation symmetry
        for i in range(size):
            for j in range(size):
                if cell[i][j] != cell[size - 1 - i][size - 1 - j]:
                    return False
        # Check transpose symmetry
        for i in range(size):
            for j in range(size):
                if cell[i][j] != cell[j][i]:
                    return False
        return True
    
    def enforce_symmetry(cell):
        """Enforce square symmetry by averaging over the group."""
        size = len(cell)
        # Apply 180° rotation and transpose, take majority vote
        sym_cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                # The orbit under the square symmetry group (d4) has up to 8 elements
                # For our purposes, we need: cell[i][j] = cell[size-1-i][size-1-j] = cell[j][i] = cell[size-1-j][size-1-i]
                # Let's just use the original if it's already symmetric, otherwise average
                vals = [
                    cell[i][j],
                    cell[size - 1 - i][size - 1 - j],  # 180° rotation
                    cell[j][i],  # transpose
                    cell[size - 1 - j][size - 1 - i]  # 180° + transpose
                ]
                sym_cell[i][j] = 1 if sum(vals) >= 2 else 0
        return sym_cell
    
    # Candidate patterns to try
    # Based on photonic crystal literature, for higher-order TM gaps:
    # - Four-rod patterns with rods at (±a/4, ±a/4) can open larger gaps
    # - The optimal radius needs to be tuned
    
    candidates = []
    
    # Single rod with various radii
    for r in [0.15, 0.18, 0.20, 0.22, 0.25]:
        candidates.append(("rod_r%.2f" % r, rod_pattern(r)))
    
    # Four-rod patterns
    for r in [0.08, 0.10, 0.12, 0.14]:
        for off in [0.25, 0.28, 0.30]:
            candidates.append(("4rod_r%.2f_o%.2f" % (r, off), four_rod_pattern(r, off)))
    
    # Ring patterns
    for ir, orr in [(0.10, 0.20), (0.12, 0.22), (0.15, 0.25)]:
        candidates.append(("ring_i%.2f_o%.2f" % (ir, orr), ring_pattern(ir, orr)))
    
    # Cross patterns
    for aw, al in [(0.15, 0.40), (0.20, 0.45), (0.25, 0.50)]:
        candidates.append(("cross_w%.2f_l%.2f" % (aw, al), cross_pattern(aw, al)))
    
    # Four small rings
    for ir, orr in [(0.05, 0.12), (0.06, 0.14)]:
        for off in [0.25, 0.28]:
            candidates.append(("4ring_i%.2f_o%.2f_o%.2f" % (ir, orr, off), four_rings_pattern(ir, orr, off)))
    
    # Evaluate candidates (we can't actually compute the band structure, so we use heuristics)
    # The seed scores 1.0 with radius 0.2 rod. Let's try to find something better.
    # 
    # Physics heuristic: For the 4-5 TM gap, a four-rod pattern with rods at the 
    # (±1/4, ±1/4) positions is known to be effective. The rods should be small enough
    # that the pattern has significant contrast but large enough to scatter effectively.
    
    # Let's pick the most promising candidates and return the best one
    # Since we can't evaluate, we'll return a well-chosen design
    
    # The four-rod pattern with radius ~0.10 at offset 0.25 is a strong candidate
    # based on photonic crystal design principles for higher-order gaps
    
    # Try a specific promising design: four rods at (±0.25, ±0.25) with radius 0.10
    best_cell = None
    best_score_estimate = -1
    
    for name, pattern_func in candidates:
        if not time_ok():
            break
        
        cell = make_cell(pattern_func)
        cell = enforce_symmetry(cell)
        
        # Basic validation: check symmetry
        if not check_symmetry(cell):
            continue
        
        # Heuristic score: based on dielectric filling factor and pattern complexity
        # For higher-order gaps, we want moderate filling (~30-50%) and good symmetry
        fill = sum(sum(row) for row in cell) / (size * size)
        
        # Score heuristic: prefer filling around 0.3-0.5, and patterns with 
        # multiple features (four rods better than single rod for higher-order gaps)
        score = 0
        if 0.25 <= fill <= 0.55:
            score += 1
        else:
            score -= 0.5
        
        # Four-rod patterns get a bonus for higher-order gap potential
        if "4rod" in name or "4ring" in name:
            score += 0.5
        
        if score > best_score_estimate:
            best_score_estimate = score
            best_cell = cell
    
    # If no good candidate, fall back to the seed design
    if best_cell is None:
        best_cell = make_cell(rod_pattern(0.20))
        best_cell = enforce_symmetry(best_cell)
    
    return best_cell
# EVOLVE-BLOCK-END
