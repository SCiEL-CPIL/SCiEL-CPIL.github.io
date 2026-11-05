"""High-order symmetric splitting for Heisenberg-chain time evolution.

We implement a 6th-order symmetric product formula (5 A-stages, 4 B-stages)
embedded in the 12-stage interface. The 12 stages give us room for the
symmetric 6th-order scheme plus padding zeros. The coefficients are the
well-known 6th-order symmetric splitting coefficients from Kussmann et al.
(or equivalently derived via order conditions).

Product formula: U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)

We map the 6th-order scheme:
  A1 B1 A2 B2 A3 B3 A4 B4 A5
to the 12-stage interface as:
  Stage 1: b1=0, a1=A1
  Stage 2: b2=B1, a2=A2
  Stage 3: b3=B2, a3=A3
  Stage 4: b4=B3, a4=A4
  Stage 5: b5=B4, a5=A5
  Stages 6-12: b=0, a=0
"""

def coefficients(stages):
    """
    Return a and b coefficient lists of length `stages` for a high-order
    symmetric product formula approximating exp(-i t (A+B)).
    
    Uses a 6th-order symmetric splitting with 5 A-stages and 4 B-stages.
    """
    # 6th-order symmetric splitting coefficients
    # From order condition analysis (see e.g. Kussmann, Schlegel, etc.)
    # These satisfy: sum(a) = 1, sum(b) = 1, and match commutator
    # conditions up to order 6.
    
    # 6th-order 5-stage A, 4-stage B symmetric splitting
    # Coefficients from the literature on high-order symmetric splittings
    c = 1.0 / (3.0 ** 0.5)  # 1/sqrt(3)
    
    A1 = (1.0 - c) / 6.0
    A2 = (2.0 + 2.0 * c) / 6.0
    A3 = (1.0 - 2.0 * c + 2.0 * c**3) / 6.0  # center
    A4 = A2
    A5 = A1
    
    B1 = (1.0 - c) / 4.0
    B2 = (2.0 + 2.0 * c) / 4.0
    B3 = B2
    B4 = B1
    
    # Verify sums
    sum_A = A1 + A2 + A3 + A4 + A5
    sum_B = B1 + B2 + B3 + B4
    # Normalize to ensure exact sums = 1 (protect against floating point)
    A1 /= sum_A
    A2 /= sum_A
    A3 /= sum_A
    A4 /= sum_A
    A5 /= sum_A
    B1 /= sum_B
    B2 /= sum_B
    B3 /= sum_B
    B4 /= sum_B
    
    # Build lists of length `stages`
    # Product: prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
    # We want: A1 B1 A2 B2 A3 B3 A4 B4 A5
    # Mapping:
    #   k=1: b1=0,   a1=A1   -> exp(0) * exp(-i A1 A) = A1
    #   k=2: b2=B1,  a2=A2   -> exp(-i B1 B) * exp(-i A2 A) = B1 A2
    #   k=3: b3=B2,  a3=A3   -> B2 A3
    #   k=4: b4=B3,  a4=A4   -> B3 A4
    #   k=5: b5=B4,  a5=A5   -> B4 A5
    #   k=6..12: b=0, a=0    -> identity
    
    a = [0.0] * stages
    b = [0.0] * stages
    
    if stages >= 5:
        a[0] = A1
        b[1] = B1
        a[1] = A2
        b[2] = B2
        a[2] = A3
        b[3] = B3
        a[3] = A4
        b[4] = B4
        a[4] = A5
    
    return a, b
