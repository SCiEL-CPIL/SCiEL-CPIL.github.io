for (li, lj), a_c in A_in.items():
    for (br, bc, a_bc) in A_side:
        gi = br * 2 + li  # row in A (0..3)
        gj = bc * 2 + lj  # col in A (0..3)
        u[gi * m + gj] += a_c * a_bc
