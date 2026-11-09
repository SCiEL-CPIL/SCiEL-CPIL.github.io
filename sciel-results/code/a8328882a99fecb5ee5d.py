import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
        remaining = bins - item
        # Only feasible bins
        feasible = remaining >= 0
        
        # Hybrid strategy:
        # Small items: Worst Fit (preserve large bins? No, Worst Fit uses large bins. 
        # Maybe the logic is: Small items are numerous. Using Worst Fit fills the large bins 
        # gradually, reducing the chance of having a large bin that *only* a large item could use 
        # but is too fragmented? No, that's not right.)
        
        # Let's trust the "Island 7" finding: "preserve large bins for future large items".
        # This phrase usually implies **Best Fit** (use small bins, keep large ones big).
        # If Island 7 found that *Worst Fit* parameters preserved large bins, maybe my interpretation of "Worst Fit" vs "Best Fit" in the context of the parameter grid was inverted or the grid explored a different space.
        # "p up to 20, b up to 2".
        
        # Let's try a pure **Best Fit** with a small penalty for *too* much empty space?
        # No, Best Fit is the seed.
        
        # Let's try **Worst Fit**.
        # If I return `remaining`, I pick the bin with the most space left.
        
        # Let's try a mix: 50% BF, 50% WF.
        # Score = 0
        # This is equivalent to random choice among feasible bins? No.
        # BF: max(-remaining). WF: max(remaining).
        # Mix: max(0 * remaining). This is bad.
        
        # Let's use the threshold approach.
        # If item is small, use Worst Fit.
        # If item is large, use Best Fit.
        
        # Why?
        # If I have many small items, and I use Best Fit, I pack them into small bins.
        # If I use Worst Fit, I spread them into large bins.
        # If a large item comes, it needs a large bin.
        # With BF, large bins are untouched.
        # With WF, large bins are fragmented.
        # So BF should be better for preserving large bins.
        
        # UNLESS... the "large items" are not that large, and the "small items" are so small that they can fit in the "fragmented" space of a large bin?
        
        # Let's look at the score: 0.5463 -> 0.6408.
        # This is a significant improvement.
        
        # I will bet on **Worst Fit** being better for this specific Weibull distribution.
        # I will implement **Worst Fit** for items < 20 and **Best Fit** for items >= 20.
        
        scores = np.zeros_like(bins)
        scores[~feasible] = -np.inf
        
        # For feasible bins:
        # If item is small, prefer large remaining space (Worst Fit).
        # If item is large, prefer small remaining space (Best Fit).
        
        # Blend factor
        # w = 1.0 for item < 10 (Pure WF)
        # w = 0.0 for item > 30 (Pure BF)
        # Linear in between.
        
        # Score = (1-w) * (bins - item) + w * (-(bins - item))
        # Score = (1 - 2w) * (bins - item)
        
        # If w=0 (small item), Score = remaining (WF).
        # If w=1 (large item), Score = -remaining (BF).
        
        # Let's set the transition between 10 and 30.
        
        w = np.clip((item - 10) / (30 - 10), 0.0, 1.0)
        coeff = 1.0 - 2.0 * w
        
        scores[feasible] = coeff * remaining[feasible]
        
        return scores
