import random

def construct():
    N = 256
    # Generate all 4-APs
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            # 1-based indices in problem, 0-based in code
            # a, a+d, a+2d, a+3d
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    # Initialize with random coloring
    random.seed(42)
    colors = [random.randint(0, 1) for _ in range(N)]
    
    def calculate_score(colors):
        count = 0
        for ap in aps:
            c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if c1 == c2 == c3 == c4:
                count += 1
        return count

    current_score = calculate_score(colors)
    
    # Local search
    improved = True
    while improved:
        improved = False
        # Calculate N0 and N1
        N0 = [0] * N
        N1 = [0] * N
        
        for ap in aps:
            i1, i2, i3, i4 = ap
            c1, c2, c3, c4 = colors[i1], colors[i2], colors[i3], colors[i4]
            
            # Check for potential monochromatic 0s if we flip one
            # If others are 0, this AP is "at risk" of being mono 0 or becomes mono 0 if flipped to 0?
            # We need N0[i]: count of APs containing i where OTHERS are 0.
            # If c2==c3==c4==0, then AP is mono 0 if c1==0.
            # If we flip c1 to 1, it breaks. If we flip c1 to 0 (if it was 1), it forms.
            # So for index i1, the number of APs where others are 0 is:
            if c2 == 0 and c3 == 0 and c4 == 0:
                N0[i1] += 1
            if c1 == 0 and c3 == 0 and c4 == 0:
                N0[i2] += 1
            if c1 == 0 and c2 == 0 and c4 == 0:
                N0[i3] += 1
            if c1 == 0 and c2 == 0 and c3 == 0:
                N0[i4] += 1
                
            if c2 == 1 and c3 == 1 and c4 == 1:
                N1[i1] += 1
            if c1 == 1 and c3 == 1 and c4 == 1:
                N1[i2] += 1
            if c1 == 1 and c2 == 1 and c4 == 1:
                N1[i3] += 1
            if c1 == 1 and c2 == 1 and c3 == 1:
                N1[i4] += 1
        
        # Find best flip
        best_delta = 0
        best_idx = -1
        for i in range(N):
            if colors[i] == 0:
                # Flip to 1
                # Lose N0[i] (mono 0s broken), Gain N1[i] (mono 1s formed)
                delta = N1[i] - N0[i]
            else:
                # Flip to 0
                # Lose N1[i] (mono 1s broken), Gain N0[i] (mono 0s formed)
                delta = N0[i] - N1[i]
            
            if delta < best_delta:
                best_delta = delta
                best_idx = i
        
        if best_idx != -1 and best_delta < 0:
            colors[best_idx] = 1 - colors[best_idx]
            current_score += best_delta
            improved = True

    return colors
