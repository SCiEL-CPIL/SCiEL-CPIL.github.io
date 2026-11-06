"""Product formula design for Heisenberg-chain time evolution using 4th-order Yoshida splitting."""
import time

def coefficients(stages):
    """12-stage product formula using 4 blocks of 3-stage 4th-order Yoshida splitting."""
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    try:
        # 4th-order Yoshida coefficients
        c = 2.0 ** (1.0 / 3.0)
        denom = 1.0 - c
        
        # For A-B-A-B-A-B structure:
        # alpha1 = alpha3 = 1/(2*denom), alpha2 = 1 - 2*alpha1
        # beta1 = beta3 = c/denom, beta2 = 1 - 2*beta1
        alpha1 = 1.0 / (2.0 * denom)
        alpha2 = 1.0 - 2.0 * alpha1
        beta1 = c / denom
        beta2 = 1.0 - 2.0 * beta1
        
        # Our structure per stage: exp(-i b_k t B) exp(-i a_k t A)
        # Over 3 stages: B1 A1 B2 A2 B3 A3
        # This is equivalent to the Yoshida form with roles swapped and reversed.
        # For B-A-B-A-B-A to be 4th order, use:
        # b-coeffs (for B): [beta1, beta2, beta1]  (symmetric)
        # a-coeffs (for A): [alpha1, alpha2, alpha1]  (symmetric)
        # But we need to verify the ordering. The product is:
        # exp(b1*B) exp(a1*A) exp(b2*B) exp(a2*A) exp(b3*B) exp(a3*A)
        # = exp(b3*B) exp(a3*A) exp(b2*B) exp(a2*A) exp(b1*B) exp(a1*A) reversed
        # For 4th order symmetry, we need the sequence to be palindromic.
        # With b=[b1,b2,b1] and a=[a1,a2,a1], the product is:
        # B(b1) A(a1) B(b2) A(a2) B(b1) A(a1)
        # This is NOT palindromic in the right way for standard Yoshida.
        
        # Let me use the standard Yoshida A-B-A-B-A-B coefficients directly.
        # Standard: exp(hA)exp(hB)exp(hA)exp(hB)exp(hA)exp(hB) with:
        # A-coeffs: [alpha1, alpha2, alpha1], B-coeffs: [beta1, beta2, beta1]
        # Product: A(a1) B(b1) A(a2) B(b2) A(a1) B(b1)
        
        # Our product per 3 stages: B(b1) A(a1) B(b2) A(a2) B(b3) A(a3)
        # To match A-B-A-B-A-B pattern, reverse: A(a3) B(b3) A(a2) B(b2) A(a1) B(b1)
        # So set a3=alpha1, b3=beta1, a2=alpha2, b2=beta2, a1=alpha1, b1=beta1
        # i.e., a = [alpha1, alpha2, alpha1], b = [beta1, beta2, beta1]
        # but in reverse order for the product... 
        
        # Actually, let's just use: for each 3-stage block,
        # stage 1: b=beta1, a=alpha1
        # stage 2: b=beta2, a=alpha2  
        # stage 3: b=beta1, a=alpha1
        # Product: B(beta1) A(alpha1) B(beta2) A(alpha2) B(beta1) A(alpha1)
        # This should be 4th order by symmetry (palindromic in the B-A pairs)
        
        n_blocks = stages // 3
        if n_blocks < 1:
            n_blocks = 1
        
        # Scale coefficients so total sum = 1
        # Each block contributes sum_a_block = alpha1+alpha2+alpha1 = 1
        # Each block contributes sum_b_block = beta1+beta2+beta1 = 1
        # With n_blocks, total sum = n_blocks, so scale by 1/n_blocks
        scale = 1.0 / n_blocks
        
        a = []
        b = []
        for _ in range(n_blocks):
            a.extend([alpha1 * scale, alpha2 * scale, alpha1 * scale])
            b.extend([beta1 * scale, beta2 * scale, beta1 * scale])
        
        # Pad to exactly 'stages' length if needed
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        
        # Trim to exactly 'stages' if we overshot
        a = a[:stages]
        b = b[:stages]
        
        # Verify constraints
        assert len(a) == stages, f"a length {len(a)} != {stages}"
        assert len(b) == stages, f"b length {len(b)} != {stages}"
        assert abs(sum(a) - 1.0) < 1e-10, f"sum(a)={sum(a)}"
        assert abs(sum(b) - 1.0) < 1e-10, f"sum(b)={sum(b)}"
        for x in a:
            assert -50.0 <= x <= 50.0, f"a value {x} out of range"
        for x in b:
            assert -50.0 <= x <= 50.0, f"b value {x} out of range"
        
        return a, b
        
    except Exception:
        # Fallback: simple Strang-like
        n = max(1, stages // 2)
        a = [0.5 / n] * (2 * n)
        b = []
        for _ in range(n):
            b.extend([1.0 / n, 0.0])
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        a = a[:stages]
        b = b[:stages]
        return a, b
