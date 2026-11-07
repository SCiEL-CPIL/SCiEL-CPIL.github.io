"""
Root-tracking optimization for the Uncertainty Inequality task.
Strategy:
1. Analyze the structure of q(x) = g(x)/x^2.
2. The value is r^2 / (2*pi) where r is the largest real root where q changes sign.
3. To minimize r, we want the largest root to be as small as possible.
4. Often, the optimal solution occurs when the largest root is a double root (tangent to the axis) 
   or when the leading coefficient is small, pushing roots inward.
5. We formulate a constrained optimization:
   Minimize r subject to q(r) = 0 and q(x) >= 0 for all x in [0, r] (or similar conditions).
   Actually, since q must change sign at r, and we want the LARGEST such root, 
   it implies q(x) does not change sign for x > r (until it goes to -inf or +inf).
   
   Let's look at the leading term. g(x) ~ c3 H12(x). 
   H12(x) ~ 12! x^12. So g(x) ~ c3 * 12! * x^12.
   q(x) = g(x)/x^2 ~ c3 * 12! * x^10.
   If c3 > 0, q(x) -> +inf as x -> inf.
   If c3 < 0, q(x) -> -inf as x -> inf.
   
   The task says: "If g tends to a negative limit at infinity, g is replaced by -g."
   This effectively means we can assume c3 > 0 for the final evaluation, or we just need to handle the sign.
   Wait, "g tends to a negative limit" -> replace by -g.
   H12(x) leading coeff is 12! > 0. So sign of g at inf is sign of c3.
   If c3 < 0, g -> -inf. Then -g is used. -g has leading coeff -c3 > 0.
   So effectively, we can assume the leading coefficient of the effective polynomial is positive.
   
   If effective leading coeff is positive, q(x) -> +inf as x -> inf.
   For q to have a largest sign-changing root r, q must come down from +inf (at -inf? no, q is even) 
   Wait, q(x) is even. q(-x) = q(x).
   Roots come in pairs +/- r.
   The "largest real root" usually implies magnitude or the positive one? 
   "r be the largest real root". Usually means the maximum value. Since roots are symmetric, 
   the largest is the positive one with max magnitude.
   
   If q(x) -> +inf as x -> inf, and it has a largest sign-changing root r, 
   then for x > r, q(x) must be positive (since it ends up positive and doesn't cross again).
   Just before r, q(x) must be negative (to cross to positive).
   So q(r) = 0, q'(r) > 0 (crossing upwards).
   
   To minimize r, we want to "push" the root to the left.
   A common structural feature in such polynomial bounds is that the optimal polynomial 
   touches the axis at the critical point (double root) or has specific symmetry.
   
   Let's try a direct parameterization.
   We have 3 free parameters c0, c1, c2. c3 is determined by g(0)=0.
   
   Let's use scipy.optimize to minimize r directly.
   We can express r as a function of c0, c1, c2 by finding the roots.
   But root finding is discontinuous.
   
   Alternative: 
   Assume the largest root r is a double root (tangent). This is a common case for optimal bounds 
   (like in Hermite interpolation or Chebyshev bounds). 
   If r is a double root, then q(r) = 0 and q'(r) = 0.
   This gives us 2 equations. We have 3 unknowns (c0, c1, c2) and r is also a variable.
   So we have 4 variables and 2 equations. We can minimize r subject to these constraints.
   
   Let's define the variables as [c0, c1, c2, r].
   Constraints:
   1. q(r) = 0
   2. q'(r) = 0
   3. q(x) >= 0 for all x in [0, r] ? 
      Actually, if r is the largest sign-changing root, and q(x)->+inf, 
      then q(x) > 0 for x > r.
      For x < r, q(x) can be anything, as long as there are no larger sign-changing roots.
      If r is a double root, it doesn't change sign. 
      Wait, the task says "at which q changes sign".
      So r MUST be a simple root (odd multiplicity).
      Therefore, r cannot be a double root!
      
   Correction: 
   If r is a double root, it does NOT change sign. 
   The task requires r to be a root where q CHANGES sign.
   So r must be a simple root.
   
   So we cannot use q'(r)=0.
   
   However, we can use the fact that there are no other sign-changing roots larger than r.
   This implies that in the interval (r, infinity), q(x) does not cross zero.
   Since q(x) -> +inf, q(x) > 0 for x > r.
   
   So the conditions are:
   1. q(r) = 0
   2. q(x) > 0 for all x > r
   3. Minimize r.
   
   This is a semi-infinite constraint.
   
   We can approximate this by sampling.
   
   Let's use `scipy.optimize.minimize` with SLSQP.
   Variables: c0, c1, c2.
   Objective: r(c0, c1, c2)
   Constraint: min_{x > r} q(x) > 0 (approximated by grid).
   
   But r depends on c.
   
   Let's try a different approach.
   Since we have 3 parameters, and the "best" case might be when the polynomial is "flattest" 
   near the root or has some other structural property.
   
   Let's look at the seed: [-1.13, 0.04, 0.000308...].
   Let's compute the value for the seed.
   
   Actually, let's just use a robust root-finding optimizer.
   We will define the objective as the value computed by the evaluator.
   But we will use a smarter initialization and a more targeted search.
   
   Key Insight: 
   The value is determined by the LARGEST root.
   If we can make the leading coefficient c3 very small, the roots might shrink?
   No, c3 is tied to c0, c1, c2.
   
   Let's try to search for c0, c1, c2 such that the largest root is minimized.
   We will use a grid search to find a good starting point, then Nelder-Mead.
   
   To ensure we find the "change of sign" root correctly, we must be careful with root multiplicity.
   
   Let's implement a function that, given c0, c1, c2, returns the value.
   Then we use differential_evolution to find a good candidate, then refine.
   
   The previous program already did this. Why did it stagnate?
   Maybe the landscape is tricky.
   
   Let's try a different representation.
   Instead of c0, c1, c2, let's parameterize the roots of q(x)?
   q(x) is degree 10. Too many roots.
   
   How about parameterizing c0, c1, c2 in a way that ensures c3 > 0?
   
   Let's try a simple but effective strategy:
   1. Use `scipy.optimize.differential_evolution` with a larger population and more iterations.
   2. Use a custom objective that penalizes invalid states (where no sign-changing root is found or g(0) != 0 check fails, though g(0)=0 is enforced by c3).
   3. After DE, use `minimize` with 'Powell' method (derivative-free, good for non-smooth).
   
   The previous program used Nelder-Mead and BFGS. Powell is often more robust for non-smooth objectives.
   
   Also, the previous program had a fixed budget. We will manage time carefully.
   
   One specific structural change: 
   The previous program checked `abs(x0_coeff) > 1e-10` and returned inf. 
   This is correct because q(x) must divide x^2 exactly, so the constant term of q must be 0.
   Wait, q(x) = g(x)/x^2.
   g(x) = c0 H0 + c1 H4 + c2 H8 + c3 H12.
   H0 = 1.
   H4(x) = x^4 - 6x^2 + 3.
   H8(x) = x^8 - 28x^6 + 280x^4 - 840x^2 + 105.
   H12(x) = x^12 - 84x^10 + 2520x^8 - 30240x^6 + 16800x^4 - 30240x^2 + 945.
   
   g(0) = c0*1 + c1*3 + c2*105 + c3*945 = 0.
   So c3 = -(c0 + 3c1 + 105c2)/945.
   
   q(x) = g(x)/x^2.
   For q(x) to be a polynomial, g(x) must be divisible by x^2.
   This means the constant term and the x^1 term of g(x) must be 0.
   Since g is even, the x^1 term is always 0.
   The constant term is g(0).
   So the condition `g(0)=0` is exactly the condition that `x^2` divides `g(x)`.
   
   So the check `abs(x0_coeff) > 1e-10` in the previous code was checking if the constant term of q is 0?
   Let's check the code:
   `x0_coeff = -30240*c3 - 840*c2 - 6*c1`
   This looks like the constant term of q?
   q(x) = c3 x^10 ... + (constant term of g)/x^2 ? No.
   g(x) = ... + g(0).
   q(x) = g(x)/x^2.
   If g(0) != 0, q(x) has a 1/x^2 singularity.
   So g(0) MUST be 0.
   The code calculates c3 to make g(0)=0.
   Then it checks `x0_coeff`. 
   Let's re-verify the expansion.
   
   H0 = 1
   H4 = x^4 - 6x^2 + 3
   H8 = x^8 - 28x^6 + 280x^4 - 840x^2 + 105
   H12 = x^12 - 84x^10 + 2520x^8 - 30240x^6 + 16800x^4 - 30240x^2 + 945
   
   g(x) = c0(1) + c1(x^4 - 6x^2 + 3) + c2(x^8 - 28x^6 + 280x^4 - 840x^2 + 105) + c3(x^12 - 84x^10 + 2520x^8 - 30240x^6 + 16800x^4 - 30240x^2 + 945)
   
   Coeff of x^2 in g(x):
   -6 c1 - 840 c2 - 30240 c3.
   
   Coeff of x^0 in g(x):
   c0 + 3 c1 + 105 c2 + 945 c3.
   
   For q(x) = g(x)/x^2 to be a polynomial, the coeff of x^0 in g(x) must be 0.
   So c0 + 3 c1 + 105 c2 + 945 c3 = 0.
   
   The previous code used:
   `c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0`
   H0_0 = 1
   H4_0 = 3 (Wait, code said 12.0? Let's check Hermite poly values at 0).
   
   Hermite Polynomials (Physicists):
   H_0(x) = 1
   H_1(x) = 2x
   H_2(x) = 4x^2 - 2
   H_3(x) = 8x^3 - 12x
   H_4(x) = 16x^4 - 48x^2 + 12
   H_8(x) = ...
   
   Let's check H_4(0).
   H_4(x) = 16x^4 - 48x^2 + 12.
   H_4(0) = 12.
   
   My manual expansion used "Probabilists" or a different scaling?
   The task specifies `sympy.polys.orthopolys.hermite_poly`.
   By default, `hermite_poly(n, x)` returns the **Physicists'** Hermite polynomial $H_n(x)$.
   
   Let's verify H_4(0) for Physicists' Hermite.
   Recurrence: $H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)$.
   H_0 = 1
   H_1 = 2x
   H_2 = 4x^2 - 2
   H_3 = 8x^3 - 12x
   H_4 = 2x(8x^3 - 12x) - 6(4x^2 - 2) = 16x^4 - 24x^2 - 24x^2 + 12 = 16x^4 - 48x^2 + 12.
   H_4(0) = 12.
   
   So the code's `H4_0 = 12.0` is correct.
   
   Now let's check H_8(0).
   H_5 = 2x H_4 - 8 H_3 = 2x(16x^4 - 48x^2 + 12) - 8(8x^3 - 12x) = 32x^5 - 96x^3 + 24x - 64x^3 + 96x = 32x^5 - 160x^3 + 120x.
   H_6 = 2x H_5 - 10 H_4 = 2x(32x^5 - 160x^3 + 120x) - 10(16x^4 - 48x^2 + 12) = 64x^6 - 320x^4 + 240x^2 - 160x^4 + 480x^2 - 120 = 64x^6 - 480x^4 + 720x^2 - 120.
   H_7 = 2x H_6 - 12 H_5 = ... (odd, H_7(0)=0)
   H_8 = 2x H_7 - 14 H_6.
   H_8(0) = -14 H_6(0).
   H_6(0) = -120.
   H_8(0) = -14 * (-120) = 1680.
   
   The code had `H8_0 = 1344.0`. This is WRONG.
   1344 is H_8(0) for Probabilists' Hermite? Or maybe I made a mistake.
   Let's re-calculate H_6(0).
   H_2(0) = -2.
   H_4(0) = 12.
   H_6(0) = 2*0*H_5(0) - 10*H_4(0) = -10*12 = -120.
   H_8(0) = 2*0*H_7(0) - 14*H_6(0) = -14*(-120) = 1680.
   
   So `H8_0` should be 1680.
   The code used 1344.
   This implies the code was calculating `c3` incorrectly based on the wrong constant term of H8.
   Consequently, the condition `g(0)=0` was not satisfied in the code's internal model, OR the code's `x0_coeff` check was compensating?
   
   Let's check the code's `x0_coeff`:
   `x0_coeff = -30240*c3 - 840*c2 - 6*c1`
   This looks like the coefficient of x^2 in g(x)?
   Coeff of x^2 in g(x) = -6 c1 - 840 c2 - 30240 c3.
   Yes.
   The code checks if this is 0.
   Why?
   If q(x) = g(x)/x^2 is a polynomial, g(x) must be divisible by x^2.
   This requires g(0)=0 AND g'(0)=0.
   Since g is even, g'(0)=0 is always true.
   So only g(0)=0 is required.
   
   However, the code checks the x^2 coefficient of g(x) (which is the constant term of q(x)).
   Wait.
   q(x) = g(x)/x^2.
   g(x) = a_2 x^2 + a_4 x^4 + ...
   q(x) = a_2 + a_4 x^2 + ...
   The constant term of q(x) is the coefficient of x^2 in g(x).
   
   Does q(x) need to have a constant term of 0?
   The task says: "Let q = g / x^2, which must divide exactly."
   It does NOT say q(0)=0.
   It says g/x^2 is a polynomial.
   
   So the check `abs(x0_coeff) > 1e-10` in the previous code is **WRONG**.
   It forces the constant term of q(x) to be 0.
   This is an extra constraint that is not in the task description!
   
   If the previous code was imposing an extra constraint, it was exploring a smaller subset of valid solutions, potentially missing the optimal one.
   
   Let's verify the task text:
   "Let q = g / x^2, which must divide exactly."
   This means g is divisible by x^2.
   It does not imply q is divisible by x^
   
   So, the strategy change is: **Remove the incorrect constraint that q(0)=0.**
   And **Fix the value of H8(0) to 1680.**
   
   This is a structural change in the problem modeling.
   
   Let's implement the corrected evaluation and optimization.
"""

import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _compute_value(c0, c1, c2):
    """Compute the objective value for given coefficients."""
    try:
        # Correct values for Physicists' Hermite polynomials at 0
        H0_0 = 1.0
        H4_0 = 12.0
        H8_0 = 1680.0
        H12_0 = 209088.0 # Let's verify H12(0)
        
        # H12(0) calculation:
        # H10(0) = -16 * H8(0) = -16 * 1680 = -26880
        # H12(0) = -20 * H10(0) = -20 * (-26880) = 537600
        # Wait, recurrence is H_{n+1} = 2x H_n - 2n H_{n-1}.
        # At x=0: H_{n+1}(0) = -2n H_{n-1}(0).
        # H0(0) = 1
        # H1(0) = 0
        # H2(0) = -2(1)H0(0) = -2
        # H3(0) = 0
        # H4(0) = -2(3)H2(0) = -6(-2) = 12
        # H5(0) = 0
        # H6(0) = -2(5)H4(0) = -10(12) = -120
        # H7(0) = 0
        # H8(0) = -2(7)H6(0) = -14(-120) = 1680
        # H9(0) = 0
        # H10(0) = -2(9)H8(0) = -18(1680) = -30240
        # H11(0) = 0
        # H12(0) = -2(11)H10(0) = -22(-30240) = 665280
        
        # The code used 209088.0. This is also WRONG.
        # 209088 / 30240 = 6.91...
        # Let's check 665280.
        
        H12_0 = 665280.0

        # Enforce g(0) = 0
        # c0*1 + c1*12 + c2*1680 + c3*665280 = 0
        c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0

        # Construct q(x) coefficients
        # g(x) = c0 H0 + c1 H4 + c2 H8 + c3 H12
        # We need the coefficients of x^2, x^4, ..., x^12 in g(x)
        # Then q(x) = g(x)/x^2 will have coefficients of x^0, x^2, ..., x^10
        
        # H0 = 1
        # H4 = 16x^4 - 48x^2 + 12
        # H8 = 256x^8 - 4480x^6 + 26880x^4 - 67200x^2 + 1680
        #      (Check: H8(0)=1680. Coeff x^2: -67200. Let's verify H8 coeff x^2)
        #      H6 = 64x^6 - 480x^4 + 720x^2 - 120
        #      H8 = 2x H7 - 14 H6.
        #      H7 = 2x H6 - 12 H5 = 2x(64x^6 - 480x^4 + 720x^2 - 120) - 12(32x^5 - 160x^3 + 120x)
        #         = 128x^7 - 960x^5 + 1440x^3 - 240x - 384x^5 + 1920x^3 - 1440x
        #         = 128x^7 - 1344x^5 + 3360x^3 - 1680x
        #      H8 = 2x(128x^7 - 1344x^5 + 3360x^3 - 1680x) - 14(64x^6 - 480x^4 + 720x^2 - 120)
        #         = 256x^8 - 2688x^6 + 6720x^4 - 3360x^2 - 896x^6 + 6720x^4 - 10080x^2 + 1680
        #         = 256x^8 - 3584x^6 + 13440x^4 - 13440x^2 + 1680
        #      So H8 coeffs:
        #      x^8: 256
        #      x^6: -3584
        #      x^4: 13440
        #      x^2: -13440
        #      x^0: 1680
        
        # H12 coefficients?
        # H10 = 2x H9 - 18 H8
        # H9 = 2x H8 - 16 H7
        #     = 2x(256x^8 - 3584x^6 + 13440x^4 - 13440x^2 + 1680) - 16(128x^7 - 1344x^5 + 3360x^3 - 1680x)
        #     = 512x^9 - 7168x^7 + 26880x^5 - 26880x^3 + 3360x - 2048x^7 + 21504x^5 - 53760x^3 + 26880x
        #     = 512x^9 - 9216x^7 + 48384x^5 - 80640x^3 + 30240x
        # H10 = 2x(512x^9 - 9216x^7 + 48384x^5 - 80640x^3 + 30240x) - 18(256x^8 - 3584x^6 + 13440x^4 - 13440x^2 + 1680)
        #     = 1024x^10 - 18432x^8 + 96768x^6 - 161280x^4 + 60480x^2 - 4608x^8 + 64512x^6 - 241920x^4 + 241920x^2 - 30240
        #     = 1024x^10 - 23040x^8 + 161280x^6 - 403200x^4 + 302400x^2 - 30240
        
        # H12 = 2x H11 - 22 H10
        # H11 = 2x H10 - 20 H9
        #      = 2x(1024x^10 - 23040x^8 + 161280x^6 - 403200x^4 + 302400x^2 - 30240) - 20(512x^9 - 9216x^7 + 48384x^5 - 80640x^3 + 30240x)
        #      = 2048x^11 - 46080x^9 + 322560x^7 - 806400x^5 + 604800x^3 - 60480x - 10240x^9 + 184320x^7 - 967680x^5 + 1612800x^3 - 604800x
        #      = 2048x^11 - 56320x^9 + 506880x^7 - 1774080x^5 + 2217600x^3 - 1209600x
        
        # H12 = 2x(2048x^11 - 56320x^9 + 506880x^7 - 1774080x^5 + 2217600x^3 - 1209600x) - 22(1024x^10 - 23040x^8 + 161280x^6 - 403200x^4 + 302400x^2 - 30240)
        #     = 4096x^12 - 112640x^10 + 1013760x^8 - 3548160x^6 + 4435200x^4 - 2419200x^2 - 22528x^10 + 506880x^8 - 3548160x^6 + 8870400x^4 - 6652800x^2 + 665280
        #     = 4096x^12 - 135168x^10 + 1520640x^8 - 7096320x^6 + 13305600x^4 - 9072000x^2 + 665280
        
        # So H12 coeffs:
        # x^12: 4096
        # x^10: -135168
        # x^8: 1520640
        # x^6: -7096320
        # x^4: 13305600
        # x^2: -9072000
        # x^0: 665280

        # Now construct g(x) coeffs for x^2, x^4, x^6, x^8, x^10, x^12
        # g(x) = c0 H0 + c1 H4 + c2 H8 + c3 H12
        
        # Coeff of x^2 in g:
        # c0*0 + c1*(-48) + c2*(-13440) + c3*(-9072000)
        # Wait, H0 has no x^2 term.
        # H4 x^2 coeff: -48
        # H8 x^2 coeff: -13440
        # H12 x^2 coeff: -9072000
        
        # Coeff of x^4 in g:
        # c0*0 + c1*(16) + c2*(13440) + c3*(13305600)
        
        # Coeff of x^6 in g:
        # c0*0 + c1*0 + c2*(-3584) + c3*(-7096320)
        
        # Coeff of x^8 in g:
        # c0*0 + c1*0 + c2*(256) + c3*(1520640)
        
        # Coeff of x^10 in g:
        # c0*0 + c1*0 + c2*0 + c3*(-135168)
        
        # Coeff of x^12 in g:
        # c0*0 + c1*0 + c2*0 + c3*(4096)
        
        # q(x) = g(x)/x^2
        # q(x) = A0 + A2 x^2 + A4 x^4 + A6 x^6 + A8 x^8 + A10 x^10
        # A0 = coeff x^2 of g
        # A2 = coeff x^4 of g
        # ...
        
        A0 = -48*c1 - 13440*c2 - 9072000*c3
        A2 = 16*c1 + 13440*c2 + 13305600*c3
        A4 = -3584*c2 - 7096320*c3
        A6 = 256*c2 + 1520640*c3
        A8 = -135168*c3
        A10 = 4096*c3
        
        # Check if c3 is effectively 0? If c3=0, A10=0.
        # If c3=0, g is degree 8. q is degree 6.
        
        # Roots of q(x)
        # q(x) = A10 x^10 + A8 x^8 + A6 x^6 + A4 x^4 + A2 x^2 + A0
        # Let y = x^2.
        # P(y) = A10 y^5 + A8 y^4 + A6 y^3 + A4 y^2 + A2 y + A0
        # Roots of P(y) are y_i.
        # Real positive roots y_i > 0 give real roots x = sqrt(y_i).
        
        coeffs = np.array([A10, A8, A6, A4, A2, A0])
        
        # Find roots of P(y)
        roots_y = np.roots(coeffs)
        
        # Filter real positive roots
        real_pos_y = roots_y[np.abs(roots_y.imag) < 1e-10].real
        real_pos_y = real_pos_y[real_pos_y > 1e-10]
        
        if len(real_pos_y) == 0:
            return float('inf')
            
        # Largest y gives largest x
        y_max = np.max(real_pos_y)
        r = np.sqrt(y_max)
        
        # Check if q changes sign at r.
        # Since we are looking for the largest root, and q(x) -> sign(A10) x^10,
        # if A10 > 0, q(x) -> +inf.
        # If A10 < 0, q(x) -> -inf.
        # The task says if g -> -inf, replace by -g.
        # So we can assume the effective leading coeff is positive.
        # If A10 < 0, we should multiply all coeffs by -1.
        
        if A10 < 0:
            coeffs = -coeffs
            A10 = -A10
            
        # Now A10 >= 0.
        # If A10 == 0, the degree is lower.
        
        # Check sign change at r.
        # Evaluate P(y) at y_max - eps and y_max + eps.
        # If they have opposite signs, it's a sign-changing root.
        
        eps = 1e-9
        p_left = np.polyval(coeffs, y_max - eps)
        p_right = np.polyval(coeffs, y_max + eps)
        
        if p_left * p_right > 0:
            # No sign change. This root might be a double root or not the "effective" boundary.
            # We need to find the LARGEST root where sign changes.
            # Sort roots
            sorted_y = np.sort(real_pos_y)[::-1]
            found = False
            for y in sorted_y:
                if y <= 0: break
                pl = np.polyval(coeffs, y - eps)
                pr = np.polyval(coeffs, y + eps)
                if pl * pr < 0:
                    r = np.sqrt(y)
                    found = True
                    break
            if not found:
                return float('inf')
        
        value = r**2 / (2 * np.pi)
        return value

    except Exception:
        return float('inf')

def _objective(params):
    """Objective function for optimization."""
    c0, c1, c2 = params
    return _compute_value(c0, c1, c2)

def construct():
    """Construct three coefficients that minimize the objective value."""
    start_time = time.time()
    deadline = start_time + 200  # Safety margin before 210s limit

    # Seed values
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_params = seed.copy()
    best_value = _compute_value(*seed)

    # Strategy 1: Differential Evolution (global optimization)
    if time.time() < deadline - 60:
        try:
            bounds = [(-5, 5), (-5, 5), (-5, 5)]
            result = differential_evolution(
                _objective, bounds, maxiter=100, popsize=20,
                tol=1e-8, mutation=(0.5, 1.0), recombination=0.7,
                seed=42, polish=True
            )
            if result.fun < best_value:
                best_params = result.x
                best_value = result.fun
        except Exception:
            pass

    # Strategy 2: Powell local refinement
    if time.time() < deadline - 30:
        try:
            result = minimize(
                _objective, best_params, method='Powell',
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if result.fun < best_value:
                best_params = result.x
                best_value = result.fun
        except Exception:
            pass

    # Strategy 3: Nelder-Mead local refinement
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_value:
                best_params = result.x
                best_value = result.fun
        except Exception:
            pass

    # Strategy 4: Fine local search around best solution
    if time.time() < deadline:
        base_value = best_value
        for scale in [1e-4, 1e-5, 1e-6, 1e-7]:
            if time.time() > deadline:
                break
            for _ in range(200):
                if time.time() > deadline:
                    break
                perturbation = np.random.normal(0, scale, size=3)
                candidate = best_params + perturbation
                val = _compute_value(*candidate)
                if val < best_value:
                    best_params = candidate
                    best_value = val

    return np.array(best_params)
