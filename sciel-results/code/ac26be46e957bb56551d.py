import numpy as np
import time

N = 600

def construct():
    # Start time for safety
    start_time = time.time()
    
    # Strategy: Two bumps at the edges of the interval [-1/4, 1/4]
    # This minimizes the peak of the autoconvolution under L1 normalization.
    # Interval [-1/4, 1/4] has length 1/2.
    # Left bump: [-1/4, -1/8] (width 1/4)
    # Right bump: [1/8, 1/4] (width 1/4)
    # Middle gap: [-1/8, 1/8] (width 1/4)
    
    # N=600 bins.
    # Each bin width dx = (1/2) / 600 = 1/1200.
    # Total width 1/2.
    # Left half is bins 0..299.
    # Right half is bins 300..599.
    
    # Left bump occupies first 1/4 of total width? No.
    # Total width 1/2.
    # Bump width 1/4.
    # So each bump is 1/2 of the total width? No.
    # 1/4 is half of 1/2.
    # So each bump takes half of the available space?
    # Left half of domain: [-1/4, 0]. Length 1/4.
    # Bump is [-1/4, -1/8]. Length 1/8.
    # This is half of the left half.
    # So bins 0..149 (150 bins).
    
    # Right half of domain: [0, 1/4]. Length 1/4.
    # Bump is [1/8, 1/4]. Length 1/8.
    # This is the second half of the right half.
    # Right half bins: 300..599.
    # Second half of these: 450..599.
    
    heights = np.zeros(N)
    
    # Calculate height
    # We assume L1 normalization: sum(heights * dx) = 1
    # dx = 1/1200
    # Number of non-zero bins = 150 + 150 = 300
    # 300 * h * (1/1200) = 1
    # 300 * h = 1200
    # h = 4
    
    # Wait, let's re-verify the previous calculation.
    # Previous: A=2.
    # Why?
    # "Total width 1/2. Left bump width 1/4."
    # If bump width is 1/4, then 2 * (1/4) * A = 1 => A = 2.
    # But here I said bump width is 1/8.
    
    # Let's check the geometry again.
    # Domain [-1/4, 1/4].
    # Left bump [-1/4, -1/8].
    # -1/4 = -3/12. -1/8 = -1.5/12.
    # Width = 1.5/12 = 1/8.
    # Yes, width is 1/8.
    
    # So A=4.
    
    # Let's check the peak conv with A=4, w=1/8.
    # Self conv peak: A^2 * w = 16 * 1/8 = 2.
    # Seed peak (L1=1, const h=2, w=1/2): A^2 * w = 4 * 1/2 = 2.
    # So this is NOT an improvement. It's the same.
    
    # To improve, we need peak < 2.
    # Peak = A^2 * w.
    # Constraint: 2 * A * w = 1 => A = 1/(2w).
    # Peak = (1/(4w^2)) * w = 1/(4w).
    # To minimize Peak, maximize w.
    # Max w is when bumps are as wide as possible without overlapping in a way that adds peaks.
    # In the previous analysis for k=2, I said max w=1/4.
    # Let's re-read my own analysis.
    # "Interval 1: [-1/4, -1/4+w]"
    # "Interval 2: [1/4-w, 1/4]"
    # Disjoint if -1/4+w <= 1/4-w => 2w <= 1/2 => w <= 1/4.
    # If w=1/4:
    # I1: [-1/4, 0].
    # I2: [0, 1/4].
    # They touch at 0.
    # If they touch, they form a single block [-1/4, 1/4].
    # This is the constant function!
    # Peak = 2.
    
    # If w < 1/4, there is a gap.
    # Peak = 1/(4w).
    # Since w < 1/4, 1/(4w) > 1.
    # But is it < 2?
    # If w is close to 1/4, e.g., w=0.24.
    # Peak = 1/(0.96) = 1.04.
    # This is < 2.
    
    # So we should choose w close to 1/4, but strictly less to avoid merging into the constant function?
    # Actually, if they touch, the convolution is the sum of the self-convs and cross-convs.
    # For the constant function, the "cross" and "self" are indistinguishable; it's one big triangle.
    # Peak is 2.
    
    # If we separate them by a tiny gap epsilon.
    # w = 1/4 - epsilon/2.
    # Peak = 1/(4w) approx 1.
    # But we must ensure the cross-convolution peak doesn't exceed this.
    # Cross conv peak height is also A^2 w = 1/(4w).
    # If the gap is small, the cross-conv (centered at 0) and self-convs (centered at +/- w/2? No) might overlap.
    
    # Self conv centers:
    # I1 center: -1/4 + w/2.
    # Self 1 center: 2*(-1/4 + w/2) = -1/2 + w.
    # I2 center: 1/4 - w/2.
    # Self 2 center: 1/2 - w.
    
    # Cross conv center: 0.
    
    # Support of Self 1: [-1/2, 0] (approx).
    # Support of Cross: [-w, w].
    # If w is close to 1/4, support of Cross is [-0.25, 0.25].
    # Support of Self 1 is [-0.5, 0].
    # They overlap in [-0.25, 0].
    # In the overlap region, the heights add up.
    # The peak of Self 1 is at -1/2+w.
    # The peak of Cross is at 0.
    # If they don't overlap at the peaks, the max is just the peak height.
    # Peak of Self 1: -1/2 + w.
    # Peak of Cross: 0.
    # Distance: w - 1/2? No.
    # If w=1/4, Peak Self 1 is -1/4. Peak Cross is 0.
    # They are separated.
    # So max is 1.
    
    # So even if they touch (w=1/4), the peaks are at -1/4 and 0.
    # Wait, if they touch, I1 is [-1/4, 0], I2 is [0, 1/4].
    # This IS the constant function on [-1/4, 1/4].
    # The autoconvolution is a single triangle on [-1/2, 1/2] with peak at 0.
    # Peak height 2.
    
    # Why did I think peaks were separated?
    # Because I treated them as separate bumps.
    # If they are contiguous, the "cross" term and "self" term merge into the main lobe.
    # The peak is at 0.
    
    # So we MUST have a gap to separate the lobes.
    # Let w = 1/4 - delta.
    # Gap size = 1/2 - 2w = 2*delta.
    # We need the gap to be large enough so that the central cross-peak (at 0) does not overlap with the side self-peaks.
    # Side self-peak position: -1/2 + w.
    # For w=1/4, pos = -1/4.
    # Cross-peak pos = 0.
    # Distance 1/4.
    # Width of each triangle is 2w = 1/2.
    # Half-width w = 1/4.
    # So the triangle extends from -1/4 to 0 (for self 1? No).
    # Self 1 support: [-1/2, 0]. Peak at -1/4.
    # Cross support: [-w, w] = [-1/4, 1/4]. Peak at 0.
    # They overlap in [-1/4, 0].
    # At x=0, Self 1 is 0. Cross is Peak.
    # At x=-1/4, Self 1 is Peak. Cross is 0.
    # The sum in between is less than Peak + Peak?
    # Actually, for triangles, the sum of two overlapping triangles of same height H and base 2w, shifted by d.
    # If d >= w, no overlap.
    # Here d = w - 1/2 + w? No.
    # Center 1: -1/2+w. Center 2: 0.
    # Distance d = 1/2 - w.
    # If 1/2 - w >= w => 1/2 >= 2w => w <= 1/4.
    # If w=1/4, d=0. They are centered at same point?
    # No.
    # If w=1/4, Center 1 = -1/2+1/4 = -1/4. Center 2 = 0.
    # Distance 1/4.
    # Half width is 1/4.
    # So they just touch at 0?
    # Self 1 ends at 0. Cross starts at -1/4.
    # They overlap.
    
    # To be safe, choose w slightly less than 1/4.
    # e.g., w = 0.24.
    
    # Bins:
    # w = 0.24.
    # Total width 0.5.
    # w is 0.24.
    # Bins per bump: 0.24 / 0.5 * 600 = 288.
    # 288 bins.
    # Height A = 1/(2*0.24) = 1/0.48 = 2.0833.
    
    # Construct:
    # Left bump: first 288 bins.
    # Right bump: last 288 bins.
    
    heights = np.zeros(N)
    w = 0.24
    num_bins = int(w / 0.5 * N)
    A = 1.0 / (2.0 * w)
    
    heights[0:num_bins] = A
    heights[N-num_bins:N] = A
    
    return list(heights)
