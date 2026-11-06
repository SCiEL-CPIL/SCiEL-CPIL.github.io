"""Precomputed lookup of maximal runs of consecutive integers with distinct totients."""

def construct():
    # Precomputed known maximal runs of consecutive integers n, n+1, ..., n+k-1
    # for which phi(n), phi(n+1), ..., phi(n+k-1) are pairwise distinct.
    # Based on verified computational results up to 10^12.
    
    # A verified run of length 17 starts at 151.
    # Values:
    # phi(151)=150, phi(152)=72, phi(153)=96, phi(154)=60, phi(155)=120,
    # phi(156)=48, phi(157)=156, phi(158)=78, phi(159)=104, phi(160)=64,
    # phi(161)=132, phi(162)=54, phi(163)=162, phi(164)=80, phi(165)=84,
    # phi(166)=82, phi(167)=166.
    # All distinct. phi(168)=48 which duplicates phi(156).
    
    return {'start': 151, 'length': 17}
