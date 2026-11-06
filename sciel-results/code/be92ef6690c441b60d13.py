import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
       scores = np.full_like(bins, -np.inf, dtype=float)
       
       # Valid bins
       valid = bins >= item
       
       if not np.any(valid):
           return scores
       
       r = bins[valid]
       
       # Harmonic Zone: item <= r < 2*item
       hz_mask = r < 2 * item
       
       if np.any(hz_mask):
           # In Harmonic Zone, prefer Largest Remaining Space (Worst-Fit)
           # Score should be high for larger r.
           # We want the highest score to correspond to max r.
           # So score = r.
           # For bins NOT in HZ (but valid), they are "too big".
           # We want Best-Fit among them, so score = -r.
           # But we must ensure HZ scores are higher than Non-HZ scores.
           # Max r in HZ is < 2*item.
           # Min r in Non-HZ is >= 2*item.
           # If we use score = r for HZ, max score < 2*item.
           # If we use score = -r for Non-HZ, max score (least negative) is -2*item.
           # Since 2*item > -2*item (for positive item), HZ scores are always higher.
           
           # Assign scores
           valid_indices = np.where(valid)[0]
           hz_indices = valid_indices[hz_mask]
           non_hz_indices = valid_indices[~hz_mask]
           
           scores[hz_indices] = bins[hz_indices]
           if len(non_hz_indices) > 0:
               scores[non_hz_indices] = -bins[non_hz_indices]
       else:
           # No bins in Harmonic Zone. All valid bins are >= 2*item.
           # Use Best-Fit: prefer smallest r.
           # Score = -r.
           valid_indices = np.where(valid)[0]
           scores[valid_indices] = -bins[valid_indices]
           
       return scores
